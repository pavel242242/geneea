import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from pathlib import Path
import os
import pathlib
import zipfile

def loadExistingIndex(persist_directory: Path, embeddings):
    vectordb = Chroma(persist_directory=str(persist_directory),
                      embedding_function=embeddings)
    return vectordb


def findAnswer(query: str):
    result = qa({"question": query, "chat_history": []})
    st.write(result["answer"])
    st.write("--")
    st.write('sources:',
             [d.metadata['source'] for d in result["source_documents"]])
    st.write("--")

def decompress_all_zips(directory_path, output_directory):
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        if file.endswith(".zip"):
            with zipfile.ZipFile(file_path, 'r') as zipf:
                zipf.extractall(output_directory)
PERSIST_PATH = "/data/in/files/chroma"
if not os.path.exists('/data/in/files/chroma'):
    os.mkdir('/data/in/files/chroma')
    zip_file_to_decompress = '/data/in/files/'
    output_directory = '/data/in/files/chroma'
    decompress_all_zips(zip_file_to_decompress, output_directory)

question = st.sidebar.text_input("question")

if os.getenv('openai_apitoken'):
    with st.sidebar:
        st.write("using openai token from env: "+ os.getenv('openai_apitoken')[0:6] + "...")
    openai_token = os.getenv('openai_apitoken')
else:
    openai_token = st.sidebar.text_input("openai token")

if st.sidebar.button("Click me") and question and openai_token:
    if openai_token:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_token)
        chatOpenAI = ChatOpenAI(openai_api_key=openai_token,
                                temperature=0,
                                model_name="gpt-3.5-turbo")
        vectordb = loadExistingIndex(Path(PERSIST_PATH), embeddings)
        qa = ConversationalRetrievalChain.from_llm(
            chatOpenAI, vectordb.as_retriever(), return_source_documents=True)
        findAnswer(question)
