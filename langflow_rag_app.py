# langflow_rag_app.py

import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from tempfile import NamedTemporaryFile

# Securely load OpenAI API key from Streamlit secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Legal Document Assistant", layout="wide")
st.title("📄 AI Legal Document Assistant")

uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
query = st.text_input("Ask a question about your documents:", placeholder="e.g. What are the termination clauses in this NDA?")

if uploaded_files and query:
    documents = []

    for file in uploaded_files:
        with NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue

        docs = loader.load()
        documents.extend(docs)

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    # Embedding & Indexing
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # RAG QA
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    with st.spinner("Searching documents..."):
        response = qa_chain.run(query)

    st.success("Answer:")
    st.write(response)

    with st.expander("View source documents"):
        for i, doc in enumerate(chunks[:5]):
            st.markdown(f"**Chunk {i+1}:**")
            st.write(doc.page_content)


