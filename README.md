# 📄 AI Document Q&A using RAG

An AI-powered Document Question Answering System built using **Python, Streamlit, LangChain, FAISS, HuggingFace Embeddings, and Google Gemini**. 
Upload any PDF document and ask questions about its content. 
The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the uploaded document before generating an answer.

## 🚀 Features of the project

- 📄 Upload PDF documents
- ✂️ Automatic text chunking
- 🧠 HuggingFace sentence embeddings
- 🗂️ FAISS vector database
- 🔍 Semantic similarity search
- 🤖 Google Gemini 2.5 Flash integration
- 💬 Chat-style conversation history
- 🗑️ Clear chat option
- 📚 Retrieved source chunks
- 📄 Source page numbers
- 🎨 Modern Streamlit UI

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Sentence Transformers
- Google Gemini API
- PyPDF
- Python Dotenv

## 🏗️ Project Structure
AI-Document-RAG/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
│   └── style.css
│
├── utils/
│   ├── pdf_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_db.py
│   └── gemini.py
│
├── data/
│   └── uploaded_pdfs/
│
└── vector_store/
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/Shachindrasinghrajput/AI-Document-RAG.git
cd AI-Document-RAG
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the virtual environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt

## 🔄 How It Works

1. Upload a PDF document.
2. Extract text using PyPDFLoader.
3. Split the text into chunks.
4. Generate embeddings using HuggingFace.
5. Store embeddings in a FAISS vector database.
6. Retrieve the most relevant chunks using semantic search.
7. Send the retrieved context and question to Google Gemini.
8. Display the generated answer along with the retrieved source chunks.

## 👨‍💻 Author

**Sachin Singh**

GitHub: https://github.com/Shachindrasinghrajput


## ⭐ If you like this project

Please consider giving it a ⭐ on GitHub.
