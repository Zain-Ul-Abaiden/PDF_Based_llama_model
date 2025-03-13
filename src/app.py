import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ Missing GROQ API key! Please set GROQ_API_KEY in your environment variables.")
    st.stop()

st.title("📄 Single PDF Question Answering with Llama3 (ChatGroq)")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the questions based on the provided context only. Provide the most accurate response."),
    ("user", "<context>\n{context}\n<context>\nQuestion: {input}")
])

# File uploader for a single PDF
uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"], accept_multiple_files=False)

def process_pdf():
    if "vectors" not in st.session_state or "uploaded_file" not in st.session_state or st.session_state.uploaded_file != uploaded_file:
        if not uploaded_file:
            st.warning("⚠️ Please upload a PDF file before proceeding.")
            return
        
        st.session_state.uploaded_file = uploaded_file  # Store uploaded file state
        
        # Save file temporarily
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text from PDF
        loader = PyMuPDFLoader(temp_file_path)
        documents = loader.load()
        os.remove(temp_file_path)
        
        if not documents:
            st.error("❌ No content extracted from PDF. Please check the uploaded file.")
            return
        
        # Create embeddings
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents)
        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)

        st.success("✅ Vector Store DB is Ready!")

# Process PDF if uploaded
if uploaded_file:
    process_pdf()

# User input for question
user_query = st.text_input("🔍 Ask a question from the uploaded document")

if user_query:
    if "vectors" not in st.session_state:
        st.warning("⚠️ No document embeddings found! Upload a PDF first.")
    else:
        start_time = time.process_time()
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.invoke({'input': user_query})
        response_time = time.process_time() - start_time
        
        answer_key = "answer" if "answer" in response else "output_text"
        if answer_key in response:
            st.write(f"**📝 Response:** {response[answer_key]}")
        else:
            st.warning("🚫 No relevant answer found.")
        
        st.write(f"⏳ **Response Time:** {response_time:.2f} seconds")
        
        with st.expander("📄 Relevant Document Sections"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("----------------------------------")
                