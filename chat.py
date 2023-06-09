import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from pathlib import Path
import os
import zipfile


# Function to load an existing index
def loadExistingIndex(persist_directory: Path, embeddings):
    vectordb = Chroma(persist_directory=str(persist_directory),
                      embedding_function=embeddings)
    return vectordb


# Function to find the answer to a query
def findAnswer(query: str):
    result = qa({"question": query, "chat_history": []})
    st.write(result["answer"])
    st.write("--")
    st.write('sources:', [d.metadata['source']
             for d in result["source_documents"]])
    st.write("--")


# Function to decompress all zip files in a directory
def decompress_all_zips(directory_path, output_directory):
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        if file.endswith(".zip"):
            with zipfile.ZipFile(file_path, 'r') as zipf:
                zipf.extractall(output_directory)


PERSIST_PATH = "/data/in/files/chroma"

# Create the persist directory if it doesn't exist
if not os.path.exists(PERSIST_PATH):
    os.mkdir(PERSIST_PATH)

    # Define the paths for zip file decompression
    zip_file_to_decompress = '/data/in/files/'
    output_directory = PERSIST_PATH

    # Decompress all zip files in the directory
    decompress_all_zips(zip_file_to_decompress, output_directory)

# Get the question input from the sidebar
question = st.sidebar.text_input("Ask your docs:")

# Get the OpenAI token from the environment variable or input field
if os.getenv('openai_apitoken'):
    with st.sidebar:
        st.write("Using OpenAI token from env: " + os.getenv('openai_apitoken')[0:6] + "...")
    openai_token = os.getenv('openai_apitoken')
else:
    openai_token = st.sidebar.text_input("OpenAI token")

# Perform the question answering when the button is clicked and valid inputs are provided
if st.sidebar.button("Click me") and question and openai_token:
    if openai_token:
        # Initialize OpenAI embeddings and chat models
        embeddings = OpenAIEmbeddings(openai_api_key=openai_token)
        chatOpenAI = ChatOpenAI(openai_api_key=openai_token, temperature=0, model_name="gpt-3.5-turbo")
        
        # Load the existing index
        vectordb = loadExistingIndex(Path(PERSIST_PATH), embeddings)
        
        # Create the ConversationalRetrievalChain from the chat model and vector store
        qa = ConversationalRetrievalChain.from_llm(chatOpenAI, vectordb.as_retriever(), return_source_documents=True)
        
        # Find the answer to the question
        findAnswer(question)
