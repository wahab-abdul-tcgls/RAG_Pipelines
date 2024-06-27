from dotenv import load_dotenv
import os
import boto3
import json
import bs4
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()
TEXT_MODEL = os.getenv("TEXT_MODEL")
SERVICE_NAME = os.getenv("AWS_SERVICE_NAME")
REGION_NAME = os.getenv("AWS_REGION_NAME")
ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY")
ACCESS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

# Load the Titan Embeddings using Bedrock client.
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

# Vector Store for Vector Embeddings
from langchain_community.vectorstores.faiss import FAISS

# Load RetrievalQA from langchain as it provides a simple interface to interact with the LLM.
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

# Imports for Data Ingestion
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import Bedrock for LLM
from langchain_community.llms.bedrock import Bedrock

# Load the website
def data_ingestion():
    loader = loader=WebBaseLoader(web_paths=("https://www.aluxurytravelblog.com/2024/06/25/luxury-living-with-raffles-in-cambodia/",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("entry-content-wrap")

                     )))
    documents = loader.load()
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=250)
    docs = text_splitter.split_documents(documents)
    return docs

def load_llm():
    # Initialize the Bedrock client with the required API keys and configurations
    bedrock = boto3.client(service_name=SERVICE_NAME,
                       region_name = REGION_NAME,
                       aws_access_key_id=ACCESS_KEY_ID,
                       aws_secret_access_key=ACCESS_SECRET_KEY)

    # Define the model parameters
    model_id = TEXT_MODEL
    model_kwargs = {
        "max_tokens": 2048,
        "temperature": 0.8,
        "top_p": .9,
        "top_k": 200
    }

    # Load the LLM from Bedrock
    response = bedrock.invoke_model(
        body=json.dumps(model_kwargs),
        modelId=model_id
    )

    # Parse and return the response
    llm = json.loads(response['body'].read().decode())
    return llm

# Vector Store for Vector Embeddings
def setup_vector_store(documents):
    # Create a vector store using FAISS from the documents and the embeddings
    vector_store = FAISS.from_documents(
        documents,
        openai_embeddings,
    )
    # Save the vector store locally
    vector_store.save_local("faiss_index")



# Create a prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If the answer is not within the context knowledge, kindly state that you do not know, rather than attempting to fabricate a response.
2. If you find the answer, please craft a detailed and concise response to the question at the end. Aim for a summary of max 250 words, ensuring that your explanation is thorough.

{context}

Question: {question}
Helpful Answer:"""

# Now we use langchain PromptTemplate to create the prompt template for our LLM
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


# Create a RetrievalQA chain and invoke the LLM
def get_response(llm, vector_store, query):
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    response = retrieval_qa.invoke(query)
    return response['result']


def streamlit_ui():
    st.set_page_config("Travelblog")
    st.header("RAG implementation using AWS Bedrock and Langchain")

    user_question = st.text_input("Ask me anything from Travel blog e.g. How good is the destination")

    with st.sidebar:
        st.title("Update Or Create Vector Embeddings")

        if st.button("Update Vector Store"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                setup_vector_store(docs)
                st.success("Done")

    if st.button("Generate Response") or user_question:
        # first check if the vector store exists
        if not os.path.exists("faiss_index"):
            st.error("Please create the vector store first from the sidebar.")
            return
        if not user_question:
            st.error("Please enter a question.")
            return
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", embeddings=openai_embeddings,
                                           allow_dangerous_deserialization=True)
            llm = load_llm()
            st.write(get_response(llm, faiss_index, user_question))
            st.success("Done")

if __name__ == "__main__":
    streamlit_ui()