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
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Load GROQ API Key
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ Missing GROQ API key! Please set GROQ_API_KEY in your environment variables.")
    st.stop()

st.title("📄 PDF-based Question Answering with Llama3 (ChatGroq)")

# Initialize LLM (Llama3)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the questions based on the provided context only. Provide the most accurate response."),
    ("user", "<context>\n{context}\n<context>\nQuestion: {input}")
])

# File uploader for PDFs
uploaded_files = st.file_uploader("📤 Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Function to process uploaded PDFs and create embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        if not uploaded_files:
            st.warning("⚠️ Please upload at least one PDF file before embedding.")
            return

        # Load PDFs dynamically
        documents = []
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())  # Save file temporarily
            
            loader = PyPDFLoader(uploaded_file.name)
            documents.extend(loader.load())  # Load text from PDF

            os.remove(uploaded_file.name)  # Delete the temporary file after loading

        if not documents:
            st.error("❌ No content extracted from PDFs. Please check the uploaded files.")
            return

        # Create embeddings
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents)
        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)

        st.success("✅ Vector Store DB is Ready!")

# Button to process embeddings
if st.button("🔍 Create Document Embeddings"):
    vector_embedding()

# UI Input for question
prompt1 = st.text_input("🔍 Ask a question from the uploaded documents")

# Process query when user inputs a question
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("⚠️ No document embeddings found! Click 'Create Document Embeddings' first.")
    else:
        start_time = time.process_time()

        # Create retrieval and response chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({'input': prompt1})
        response_time = time.process_time() - start_time

        # Debugging: Check response keys (optional)
        # st.write(response.keys())  

        # Display response
        answer_key = "answer" if "answer" in response else "output_text"  # Adjust if needed
        if answer_key in response:
            st.write(f"**📝 Response:** {response[answer_key]}")
        else:
            st.warning("🚫 No relevant answer found.")

        st.write(f"⏳ **Response Time:** {response_time:.2f} seconds")

        # Display relevant document chunks
        with st.expander("📄 Relevant Document Sections"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("----------------------------------")
