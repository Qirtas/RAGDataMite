import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
logger = logging.getLogger(__name__)

def setup_retriever(persist_directory="RAG/ProcessedDocuments/chroma_db",
                    model_name="all-MiniLM-L6-v2",
                    k=3):
    """
        Loads the Chroma vector store and initializes a retriever for querying.

        Args:
            persist_directory (str): Path to the Chroma DB directory.
            model_name (str): Name of the HuggingFace embedding model.
            k (int): Number of top documents to retrieve per query.

        Returns:
            VectorStoreRetriever: A configured retriever object.
        """
    logger.info(f"Loading vector store from {persist_directory}")

    embedding_function = HuggingFaceEmbeddings(model_name=model_name)

    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embedding_function)

    logger.info(f"Vector store loaded with {vectorstore._collection.count()} documents")

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    logger.info(f"Retriever initialized to fetch top {k} documents")
    return retriever


def test_retrieval(query, retriever=None, persist_directory="RAG/ProcessedDocuments/chroma_db", k=3):
    """
        Runs a query against the vector store retriever and prints basic info about retrieved documents.

        Args:
            query (str): The query text.
            retriever (VectorStoreRetriever, optional): A retriever object to use. If None, creates one.
            persist_directory (str): Path to the Chroma DB directory. Used if retriever is None.
            k (int): Number of top documents to retrieve.

        Returns:
            list: List of retrieved Document objects.
        """
    if retriever is None:
        retriever = setup_retriever(persist_directory=persist_directory, k=k)

    logger.info(f"\nQuery: {query}")

    docs = retriever.get_relevant_documents(query)

    logger.info(f"Retrieved {len(docs)} documents")

    for i, doc in enumerate(docs):
        logger.info(f"\n--- Document {i + 1} ---")
        logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
        logger.info(f"Type: {doc.metadata.get('type', 'Unknown')}")
        logger.info(f"Name: {doc.metadata.get('name', 'Unknown')}")

        content_snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        logger.info(f"Content snippet: {content_snippet}")

    return docs


def run_test_queries(queries, k=3):
    """
        Runs multiple queries through a shared retriever.

        Args:
            queries (list): A list of query strings.
            k (int): Number of top documents to retrieve for each query.

        Returns:
            dict: A dictionary mapping each query string to its list of retrieved documents.
        """
    retriever = setup_retriever(k=k)

    results = {}
    for query in queries:
        docs = test_retrieval(query, retriever, k=k)
        results[query] = docs

    return results

#
# def evaluate_retrieval_precision(query, expected_doc_names, retriever=None, k=5):
#
#     docs = test_retrieval(query, retriever, k=k)
#
#     retrieved_names = [doc.metadata.get('name', '') for doc in docs]
#
#     hits = sum(1 for name in retrieved_names if name in expected_doc_names)
#
#     precision = hits / len(docs) if docs else 0
#     recall = hits / len(expected_doc_names) if expected_doc_names else 0
#
#     print("\nRetrieval Evaluation:")
#     print(f"Expected documents: {expected_doc_names}")
#     print(f"Retrieved documents: {retrieved_names}")
#     print(f"Precision: {precision:.2f} ({hits} hits out of {len(docs)} retrieved)")
#     print(f"Recall: {recall:.2f} ({hits} hits out of {len(expected_doc_names)} expected)")
#
#     return {"precision": precision, "recall": recall, "hits": hits,
#             "retrieved": len(docs), "expected": len(expected_doc_names)}
