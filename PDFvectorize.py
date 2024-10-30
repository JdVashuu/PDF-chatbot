import os
from PyPDF2 import PdfReader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain_openai.llms import OpenAI
from typing_extensions import Concatenate

from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from web_template import css, bot_template, user_template

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
DOC_PATH = "data.pdf"


def get_pdf_content(document):
    raw_text = ""
    pdf_reader = PdfReader(document)

    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    return raw_text

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks

def get_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_storage = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vector_storage

def start_conversation(vector_embeddings):
    llm = OpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def process_query(query_text):
    if st.session_state.conversation is None:
       st.warning("Please upload a PDF and click 'Run' first.")
       return

    docs = st.session_state.vector_store.similarity_search(query_text)
    response = st.session_state.conversation.run(input_documents=docs, question=query_text)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append({"role": "user", "content": query_text})
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)
    st.header("Welcome to PDF Chatbot")
    query = st.text_input("How can I help you today?")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    with st.sidebar:
        st.subheader("PDF documents")
        document = st.file_uploader(
            "Upload your PDF files", type=["pdf"], accept_multiple_files=False
        )

        if st.button("Run") and document is not None:
            with st.spinner("Processing..."):
                # Load PDF and prepare embeddings
                extracted_text = get_pdf_content(document)
                text_chunks = get_chunks(extracted_text)
                vector_embedding = get_embeddings(text_chunks)
                st.session_state.vector_store = vector_embedding

                # Initialize the conversation chain
                st.session_state.conversation = start_conversation(vector_embedding)
                st.success("PDF processed successfully!")

    if query:
        process_query(query)

if __name__ == "__main__":
    main()
