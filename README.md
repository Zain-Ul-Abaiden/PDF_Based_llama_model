# PDF_Based_llama_model - PDF-Based AI Chatbot with Llama3

## 📌 Prerequisites
- Python 3.10+ installed
- [Git](https://git-scm.com/downloads) installed

## 🚀 Installation Steps

### 1️⃣ Clone the Repository 

git clone https://github.com/Zain-Ul-Abaiden/PDF_Based_llama_model.git <br>
cd PDF_Based_llama_model

### 2️⃣ Create a Virtual Environment

python -m venv llama_env <br>
source llama_env/bin/activate  # On macOS/Linux <br>
llama_env\Scripts\activate     # On Windows <br>

### 3️⃣ Install Dependencies

pip install -r requirements.txt <br>

### 4️⃣ Set Up Environment Variables

Create a .env file in the PDF_Based_llama_model folder and add:

GROQ_API_KEY=your_api_key_here

### 5️⃣ Run the Streamlit App

cd src <br>
streamlit run app.py