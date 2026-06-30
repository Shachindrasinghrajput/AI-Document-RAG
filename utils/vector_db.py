import os
from langchain_community.vectorstores import FAISS


VECTOR_DB_PATH = "vector_store/faiss_index"


def create_vector_store(chunks, embeddings):
    """
    Create a FAISS vector database and save it locally.
    """

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    os.makedirs("vector_store", exist_ok=True)

    vector_store.save_local(VECTOR_DB_PATH)

    return vector_store


def load_vector_store(embeddings):
    """
    Load an existing FAISS vector database.
    """

    if not os.path.exists(VECTOR_DB_PATH):
        return None

    vector_store = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store