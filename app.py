import os
import streamlit as st

from utils.pdf_loader import load_pdf
from utils.text_splitter import split_documents
from utils.embeddings import get_embeddings
from utils.vector_db import create_vector_store, load_vector_store
from utils.gemini import get_answer

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Document Q&A",
    page_icon="📄",
    layout="wide"
)

# ---------------- LOAD CSS ---------------- #

def load_css():
    with open("assets/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()
@st.cache_resource
def load_embedding_model():
    return get_embeddings()
@st.cache_data
def process_pdf(file_path):
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    return documents, chunks
# ---------------- SESSION STATE ---------------- #

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.markdown("""
# 📄 AI Document Assistant

### 🤖 Ask questions from your PDF using AI

**Powered by LangChain • FAISS • HuggingFace • Gemini**
""")

    st.markdown("---")

    st.markdown("## 🚀 Features")

    st.markdown("""
- 📄 Upload PDF
- ✂️ Smart Chunking
- 🧠 HuggingFace Embeddings
- 🗂️ FAISS Vector Store
- 🔍 Semantic Search
- 🤖 Gemini 2.5 Flash
- 💬 Chat History
""")

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()

    st.markdown("---")

    st.info(
        "Upload a PDF and ask questions based on the uploaded document."
    )

# ---------------- MAIN ---------------- #

st.markdown("""
# 📄 AI Document Assistant

### 🤖 Ask questions from your PDF using AI

Powered by **LangChain • FAISS • HuggingFace • Gemini**
""")

uploaded_file = st.file_uploader(
    "📂 Upload your PDF",
    type=["pdf"],
    help="Upload a PDF document and ask questions from it."
)

if uploaded_file is not None:

    os.makedirs(
        "data/uploaded_pdfs",
        exist_ok=True
    )

    file_path = os.path.join(
        "data",
        "uploaded_pdfs",
        uploaded_file.name
    )

    # 👇 Add it here
    if st.session_state.current_pdf != uploaded_file.name:

        st.session_state.current_pdf = uploaded_file.name

        if "vector_store" in st.session_state:
            del st.session_state.vector_store

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.success(
        f"✅ {uploaded_file.name} uploaded successfully!"
    )

    try:

        with st.spinner("⏳ Processing PDF..."):

            documents = load_pdf(file_path)

            chunks = split_documents(documents)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("📄 Pages", len(documents))

        with col2:
            st.metric("📑 Chunks", len(chunks))

       
        embeddings = load_embedding_model()
        # Load existing vector store or create a new one
        # Create vector store only once per uploaded PDF
        if "vector_store" not in st.session_state:

            with st.spinner("🧠 Creating Vector Database..."):

                st.session_state.vector_store = create_vector_store(
                    chunks,
                    embeddings
                )

        vector_store = st.session_state.vector_store

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}
        )

        st.success("✅ PDF Processed Successfully!")

        question = st.chat_input(
            "Ask a question about your PDF..."
        )

        if question:

            with st.spinner("🤖 Gemini is thinking..."):

                relevant_docs = retriever.invoke(question)

                context = "\n\n".join(
                    [doc.page_content for doc in relevant_docs]
                )

                answer = get_answer(
                    question,
                    context
                )

            # Save conversation
            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer
                }
            )

        # Display entire chat history
        if st.session_state.chat_history:

            st.markdown("---")
            st.subheader("💬 Conversation")

            for chat in st.session_state.chat_history:

                with st.chat_message("user"):
                    st.markdown(chat["question"])

                with st.chat_message("assistant"):
                    st.markdown(chat["answer"])

        # Show retrieved chunks
        if question:

            st.markdown("---")

            with st.expander("📚 Retrieved Source Chunks"):

                for i, doc in enumerate(relevant_docs, start=1):

                    st.markdown(f"### 📄 Chunk {i}")

                    page = doc.metadata.get("page")

                    if page is not None:
                        st.caption(f"Page {page + 1}")

                    st.write(doc.page_content)

                    st.divider()

    except Exception as e:

        st.error("❌ Something went wrong while processing the PDF.")

        st.exception(e)

st.markdown("---")

st.markdown(
    """
<div class="footer">
Built with ❤️ by <b>Sachin Singh</b><br>
Python • Streamlit • LangChain • FAISS • HuggingFace • Gemini
</div>
""",
    unsafe_allow_html=True
)