# Scanned PDF Question Answering with Llama3 (ChatGroq)

This project is a **Streamlit-based web application** that allows users to upload **scanned or text-based PDFs**, extract their content, and ask questions based on the document. It uses **Llama3-8b-8192** via **Groq API** for question-answering with **RAG (Retrieval-Augmented Generation)**.

---

## 🚀 Features
- Upload **scanned** or **text-based** PDFs.
- Extract text using **OCR (Tesseract)** for scanned documents.
- Store and retrieve document embeddings using **FAISS**.
- Ask questions, and get AI-generated answers based on the document's content.
- Uses **Llama3** via **Groq API** for smart responses.

---

## 📌 Prerequisites
Make sure you have the following installed:

- **Python 3.8+**
- **pip**
- **Tesseract OCR** ([Download](https://github.com/UB-Mannheim/tesseract/wiki))
- **Poppler** (for PDF to image conversion) ([Download](https://github.com/oschwartz10612/poppler-windows/releases))
- A **Groq API Key** (Get it from [Groq API](https://groq.com))

---

## 🛠 Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/BigOsoft.git
   cd BigOsoft
   ```
2. **Create a Virtual Environment & Activate It:**
   ```bash
   python -m venv llama_env
   # Activate the environment:
   # Windows
   llama_env\Scripts\activate
   # macOS/Linux
   source llama_env/bin/activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set Up Environment Variables:**
   - Create a `.env` file in the root directory and add:
     ```ini
     GROQ_API_KEY=your_groq_api_key_here
     ```

---

## 🔄 Usage

1. **Run the Streamlit App:**
   ```bash
   streamlit run src/app.py
   ```
2. **Upload a PDF.**
3. **Ask questions** based on the document.
4. **Get AI-powered answers!**

---

## 📂 Project Structure
```
BigOsoft/
│-- llama_env/           # Virtual environment (not uploaded to GitHub)
│-- src/
│   │-- app.py          # Main application script
│-- .env                # Stores API keys (not uploaded to GitHub)
│-- .gitignore          # Git ignore settings
│-- README.md           # Project documentation
│-- requirements.txt    # Python dependencies
```

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 🤝 Contributing
Feel free to submit issues or pull requests to improve the project!

---

## 📞 Contact
- **Author:** Zain Ul Abaiden
- **GitHub:** [zain-ul-abaiden](https://github.com/zain-ul-abaiden)