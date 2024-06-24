import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from typing_extensions import Concatenate
import os
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain.callbacks import get_openai_callback


class ScriptReader:
    def __init__(self):
        self._pdf_reader = None
        self._embeddings = OpenAIEmbeddings()
        self._vector_store = None
        self._query = ""

    def _read_pdf(self, uploaded_file):
        if uploaded_file is not None:
            self._pdf_reader = PdfReader(uploaded_file)
            st.success("PDF uploaded successfully!")

    def _extract_text_from_pdf(self):
        raw_text = ''
        for i, page in enumerate(self._pdf_reader.pages):
            content = page.extract_text()
            if content:
                raw_text += content
        return raw_text

    def _split_text(self, raw_text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        return text_splitter.split_text(raw_text)

    def _create_vector_store(self, texts):
        self._vector_store = FAISS.from_texts(texts, embedding=self._embeddings)

    def _accept_user_query(self):
        self._query = st.text_input("Ask questions about your PDF file:")

    def _answer_query(self):
        if self._query:
            docs = self._vector_store.similarity_search(query=self._query, k=3)
            chain = load_qa_chain(OpenAI(), chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=self._query)
                print(cb)
            st.write(response)

    def run_script_reader(self):
        st.title("Script Reader")
        st.title("Upload a PDF Document")
        uploaded_file = st.file_uploader("Choose a PDF file:", type=["pdf"])
        self._read_pdf(uploaded_file)
        if self._pdf_reader:
            raw_text = self._extract_text_from_pdf()
            texts = self._split_text(raw_text)
            self._create_vector_store(texts)
            self._accept_user_query()
            self._answer_query()


if __name__ == '__main__':
    script_reader = ScriptReader()
    script_reader.run_script_reader()
