import streamlit as st
import os
import time
import pytesseract
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image
import fitz  # PyMuPDF for fallback rendering
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document  # Ensure correct document format

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ Missing GROQ API key! Please set GROQ_API_KEY in your environment variables.")
    st.stop()

st.title("📄 Scanned PDF Question Answering with Llama3 (ChatGroq)")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the questions based on the provided context only. Provide the most accurate response."),
    ("user", "<context>\n{context}\n<context>\nQuestion: {input}")
])

# File uploader for a single PDF
uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"], accept_multiple_files=False)

def extract_text_from_images(pdf_path):
    """Extract text from scanned PDF images using OCR.
       Try pdf2image first; if it fails, use PyMuPDF (fitz) as a fallback.
    """
    poppler_path = r"C:\poppler-24.08.0\Library\bin"
    
    try:
        # Try using pdf2image to convert PDF pages to images
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
    except Exception as e:
        st.error(f"pdf2image failed: {e}. Trying alternative extraction using PyMuPDF...")
        images = []
        try:
            # Use PyMuPDF (fitz) as fallback to render pages to images
            doc = fitz.open(pdf_path)
            for page in doc:
                pix = page.get_pixmap()
                # Convert the pixmap to a PIL image
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                images.append(img)
        except Exception as e2:
            st.error(f"Fallback extraction failed: {e2}")
            return ""
    
    extracted_text = ""
    for image in images:
        text = pytesseract.image_to_string(image)
        extracted_text += text + "\n"
    return extracted_text

def process_pdf():
    if ("vectors" not in st.session_state or "uploaded_file" not in st.session_state or 
        st.session_state.uploaded_file != uploaded_file):
        if not uploaded_file:
            st.warning("⚠️ Please upload a PDF file before proceeding.")
            return
        
        st.session_state.uploaded_file = uploaded_file  # Store uploaded file state
        
        # Save file temporarily
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text using PyMuPDFLoader first (for text-based PDFs)
        loader = PyMuPDFLoader(temp_file_path)
        documents = loader.load()
        
        # If no text was extracted, try OCR
        if not documents or all(doc.page_content.strip() == "" for doc in documents):
            st.info("🛠 No readable text found. Using OCR to extract content...")
            extracted_text = extract_text_from_images(temp_file_path)
            
            if not extracted_text.strip():
                st.error("❌ OCR could not extract text. Please upload a clearer scanned document.")
                os.remove(temp_file_path)
                return
            
            # Convert extracted OCR text into LangChain Document objects
            documents = [Document(page_content=extracted_text)]
        
        os.remove(temp_file_path)  # Cleanup temp file
        
        # Create embeddings
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents)
        
        # Ensure proper document format before using FAISS
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
