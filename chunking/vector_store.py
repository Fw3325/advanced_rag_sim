from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import torch

def load_embeddings(embedding_model="BAAI/bge-large-en-v1.5"):
    """
    Load a text embedding model from Hugging Face.

    Args:
        embedding_model (str): The name of the Hugging Face embedding model to load.

    Returns:
        HuggingFaceEmbeddings: An initialized embedding model that can convert text to vector representations.
    """
    return HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}   
    )

def create_faiss_index(documents, embedding_model, save_path):
    """
    Create a FAISS vector index from a collection of text documents.

    This function embeds the provided documents using the specified embedding model
    and builds a FAISS index for efficient similarity search.

    Args:
        documents (list): A list of document objects that can be processed by the FAISS.from_documents method.
                          Each document should have a page_content attribute containing the text to be embedded.
        embedding_model (str): The name of the Hugging Face embedding model to use.

    Returns:
        FAISS: A FAISS vector store containing the embedded documents, which can be used
               for semantic similarity search.
    """

    # Load the specified embedding model
    embeddings = load_embeddings(embedding_model)

    # Create a FAISS index from the documents using the loaded embeddings
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(save_path)
    print(f"✅ FAISS index saved to: {save_path}")

    return vectorstore

def load_faiss_index(embedding_model, save_path):
    """Load FAISS index"""
    embeddings = load_embeddings(embedding_model)

    return FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True  
    )