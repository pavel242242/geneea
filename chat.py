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


PERSIST_PATH = "/data/in/files"

root = pathlib.Path('/data')
st.write(list(root.rglob("*")))
#st.write(os.listdir(PERSIST_PATH))

def findAnswer(query: str):
    result = qa({"question": query, "chat_history": []})
    st.write(result["answer"])
    st.write("--")
    st.write('sources:',
             [d.metadata['source'] for d in result["source_documents"]])
    st.write("--")

def decompress_zip(zip_file_path, output_directory):
    with zipfile.ZipFile(zip_file_path, 'r') as zipf:
        zipf.extractall(output_directory, overwrite=True)

zip_file_to_decompress = '/data/in/files/chromaoutput.zip.zip'
output_directory = '/data/in/files/'
decompress_zip(zip_file_to_decompress, output_directory)

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
        st.write(vectordb.get())
        qa = ConversationalRetrievalChain.from_llm(
            chatOpenAI, vectordb.as_retriever(), return_source_documents=True)
        findAnswer(question)
