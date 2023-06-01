import streamlit as st

from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from git import Repo

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from pathlib import Path

import os
import shutil
import nltk

# for both indexing and querying
#OPENAI_KEY = "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"
PERSIST_PATH = "out/files"
if not os.path.exists('out'):
    os.mkdir('out')
if not os.path.exists('out/files'):
    os.mkdir('out/files')

# only for indexing
#REPOSITORY_URL = "https://github.com/keboola/connection-docs"
REPOSITORY_LOCAL_PATH = "repo"

repo_url = st.sidebar.text_input("repo url")

if os.getenv('openai_apitoken'):
    with st.sidebar:
        with st.echo():
            st.write("using openai token from env: "+ os.getenv('openai_apitoken')[0:6] + "...")
    openai_token = os.getenv('openai_apitoken')
else:
    st.sidebar.text_input("openai token")


def loadDocuments(path: Path):
    loader = DirectoryLoader(path,
                             glob="**/*.md",
                             loader_cls=UnstructuredMarkdownLoader)
    return loader.load()


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


if st.sidebar.button("Click me") and repo_url and openai_token:
    REPOSITORY_URL = repo_url
    OPENAI_KEY = openai_token
    st.write("Using")
    st.write("repo: " + REPOSITORY_URL)
    st.write("Token:" + OPENAI_KEY[0:8])
    st.write("Generating chromadb")
    if os.path.exists(REPOSITORY_LOCAL_PATH):
        shutil.rmtree(REPOSITORY_LOCAL_PATH)
    st.write("Cloning repo")
    Repo.clone_from(REPOSITORY_URL, REPOSITORY_LOCAL_PATH)
    st.write("Downloading nltk")
    nltk.data.path.append('nltk_data')
    if not os.path.exists('nltk_data'):
        os.mkdir('nltk_data')
    nltk.download('popular', download_dir='nltk_data')
    st.write("Init embedder")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    st.write("Creating index")
    createNewIndex(Path(REPOSITORY_LOCAL_PATH), Path(PERSIST_PATH), embeddings)
    st.write("Done")