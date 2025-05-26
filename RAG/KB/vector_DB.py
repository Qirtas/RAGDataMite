import os
import pickle
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def create_vectorstore(embeddings_file="RAG/ProcessedDocuments/document_embeddings.pkl",
                       persist_directory="RAG/ProcessedDocuments/chroma_db"):

    print(f"Loading document embeddings from {embeddings_file}")

    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)

    documents = data["documents"]
    model_name = data["model_name"]

    print(f"Loaded {len(documents)} documents with embeddings from model {model_name}")

    print(f"Initializing embedding model: {model_name}")
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)

    print(f"Creating Chroma vector store in {persist_directory}")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=persist_directory
    )

    # Persist the vector store
    # vectorstore.persist()

    print(f"Vector store created and saved to {persist_directory}")
    print(f"Total documents in vector store: {vectorstore._collection.count()}")

    return vectorstore
