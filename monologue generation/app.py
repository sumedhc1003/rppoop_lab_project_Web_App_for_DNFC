from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import streamlit as st

st.title("Monologue Generator")

# Gemini as chat model
# Load environment variables from the .env file
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

#openai embeddings
openai_api_key = os.environ.get('OPENAI_API_KEY')  # Assuming you get the key from somewhere else
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='monologue (2).csv', source_column="prompt")
    data = loader.load()
    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=embeddings)

    return vectordb

def get_qa_chain(vectordb):

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, suggest two monolgues word to word from the given context only.
    you can explain which one is more suitable under which circumstances.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain


if __name__ == "__main__":
    vectordb = create_vector_db()
    print("\nvector db created\n")

    query = st.text_input("What's on your mind?...")
    if query:
        chain = get_qa_chain(vectordb)
        response = chain(query)

        st.header("Results")
        st.write(response["result"])



