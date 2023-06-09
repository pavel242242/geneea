import streamlit as st
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from git import Repo
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from pathlib import Path
import os
import shutil
import nltk

# Directory paths
PERSIST_PATH = "out/files"
REPOSITORY_LOCAL_PATH = "repo"

# Create directories if they don't exist
if not os.path.exists('out'):
    os.mkdir('out')
if not os.path.exists('out/files'):
    os.mkdir('out/files')

# Function to load documents from a directory
def loadDocuments(path: Path):
    loader = DirectoryLoader(path,
                             glob="**/*.md",
                             loader_cls=UnstructuredMarkdownLoader)
    return loader.load()


# Function to create a new index
def createNewIndex(documents_path: Path, persist_path: Path, embeddings):
    data = loadDocuments(documents_path)
    st.write("-- Documents Loaded -- ")

    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    splits = text_splitter.split_documents(data)
    st.write("-- Documents Split -- ")

    vectordb = Chroma.from_documents(splits,
                                     embeddings,
                                     persist_directory=str(persist_path))
    st.write("-- Vectors Created -- ")

    vectordb.persist()
    st.write("-- Vectors Persisted -- ")

    return vectordb

# Get repository URL and OpenAI token from the sidebar
repo_url = st.sidebar.text_input("Repository URL")
openai_token = st.sidebar.text_input("OpenAI token")

# Perform indexing when the button is clicked and valid inputs are provided
if st.sidebar.button("Click me") and repo_url and openai_token:
    REPOSITORY_URL = repo_url
    OPENAI_KEY = openai_token
    st.write("Using:")
    st.write("Repository: " + REPOSITORY_URL)
    st.write("Token: " + OPENAI_KEY[0:8])
    st.write("Generating ChromaDB")

    # Clone the repository
    if os.path.exists(REPOSITORY_LOCAL_PATH):
        shutil.rmtree(REPOSITORY_LOCAL_PATH)
    st.write("Cloning repository")
    Repo.clone_from(REPOSITORY_URL, REPOSITORY_LOCAL_PATH)

    # Download NLTK data
    st.write("Downloading NLTK data")
    nltk.data.path.append('nltk_data')
    if not os.path.exists('nltk_data'):
        os.mkdir('nltk_data')
    nltk.download('popular', download_dir='nltk_data')

    # Initialize OpenAI embeddings
    st.write("Initializing embedder")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

    # Create the index
    st.write("Creating index")
    createNewIndex(Path(REPOSITORY_LOCAL_PATH), Path(PERSIST_PATH), embeddings)

    # List the files in the persist directory
    import pathlib
    root = pathlib.Path(PERSIST_PATH)
    st.write(list(root.rglob("*")))

    st.write("Done")
