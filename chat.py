import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from pathlib import Path
import os

def loadExistingIndex(persist_directory: Path, embeddings):
    vectordb = Chroma(persist_directory=str(persist_directory),
                      embedding_function=embeddings)
    return vectordb


PERSIST_PATH = "in/files"
OPENAI_KEY = openai_token




def findAnswer(query: str):
    result = qa({"question": query, "chat_history": []})
    st.write(result["answer"])
    st.write("--")
    st.write('sources:',
             [d.metadata['source'] for d in result["source_documents"]])
    st.write("--")


question = st.sidebar.text_input("question")

if os.getenv('openai_apitoken'):
    st.sidebar.text("using openai token from env: "+ os.getenv('openai_apitoken')[0:6] + "...")
    openai_token = os.getenv('openai_apitoken')
else:
    openai_token = st.sidebar.text_input("openai token")

if st.sidebar.button("Click me") and question and openai_token:
    if OPENAI_KEY:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
        chatOpenAI = ChatOpenAI(openai_api_key=OPENAI_KEY,
                                temperature=0,
                                model_name="gpt-3.5-turbo")
        vectordb = loadExistingIndex(Path(PERSIST_PATH), embeddings)

        qa = ConversationalRetrievalChain.from_llm(
            chatOpenAI, vectordb.as_retriever(), return_source_documents=True)
        findAnswer(question)
