import pandas as pd
from langchain.docstore.document import Document
from RAG.KB.preprocess_data import preprocess_data
import logging
import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from RAG.KB.generating_embeddings import generate_embeddings
from RAG.KB.vector_DB import create_vectorstore
from RAG.KB.ingest_documents import ingest_documents
from RAG.Retrieval.retriever import run_test_queries, test_retrieval

logging.basicConfig(
    level=logging.INFO,  # Can change to DEBUG when needed
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    # 1. Preprocess CSV Files to clean for JSON related issues


    output_dir = "RAG/Content/ProcessedFiles"

    try:
        processed_files = preprocess_data(output_dir)

        logger.info("\nPreprocessing Summary:")
        logger.info("=====================")
        for file_type, file_path in processed_files.items():
            logger.info(f"{file_type}: {file_path}")

        logger.info("\nTo use these files with KB.py:")
        logger.info("```python")
        logger.info("from KB import load_csv_as_documents, load_all_csvs_as_documents")
        logger.info("")
        logger.info("# Define processed file paths")
        logger.info("csv_files = {")
        for file_type, file_path in processed_files.items():
            logger.info(f"    '{file_type}': '{file_path}',")
        logger.info("}")
        logger.info("")
        logger.info("# Load all documents")
        logger.info("all_docs = load_all_csvs_as_documents(csv_files)")
        logger.info("print(f\"Loaded {len(all_docs)} total documents\")")
        logger.info("```")

    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        sys.exit(1)
    


    # 2. Documents Ingestion - converting to Langchain documents


    documents = ingest_documents()

    if documents:
        logger.info("\nSample document:")
        logger.info("-" * 80)

        sample_doc = next((doc for doc in documents if doc.metadata.get("type") == "KPI"), documents[0])

        logger.info("Content:")
        logger.info(sample_doc.page_content[:500] + "..." if len(sample_doc.page_content) > 500 else sample_doc.page_content)
        logger.info("-" * 80)
        logger.info("Metadata:")
        logger.info(sample_doc.metadata)
    

    # 3. Analysing if there is a need for text chunking

    with open('RAG/ProcessedDocuments/all_documents.pkl', 'rb') as f:
        documents = pickle.load(f)

    doc_lengths = defaultdict(list)
    for doc in documents:
        doc_type = doc.metadata.get('type', 'Unknown')
        length = len(doc.page_content)
        doc_lengths[doc_type].append(length)

    for doc_type, lengths in doc_lengths.items():
        logger.info(f"\n{doc_type} Documents:")
        logger.info(f"  Count: {len(lengths)}")
        logger.info(f"  Average length: {np.mean(lengths):.1f} characters")
        logger.info(f"  Min length: {min(lengths)} characters")
        logger.info(f"  Max length: {max(lengths)} characters")

        if lengths:
            longest_idx = np.argmax(lengths)
            longest_doc = next((doc for i, doc in enumerate(documents)
                                if doc.metadata.get('type') == doc_type
                                and i == longest_idx), None)
            if longest_doc:
                logger.info(f"  Longest document name: {longest_doc.metadata.get('name', 'Unknown')}")
                logger.info(f"  First 200 chars: {longest_doc.page_content[:200]}...")



    # 4. Generating vector embeddings from Langchain documents

    generate_embeddings()


    # 5. Creating vector DB and save embeddings to Vector DB

    create_vectorstore()

    # 6. Testing Retrieval

    # "What is Access Cost?",
    # "Explain the Financial Perspective in BSC",
    # "How do we measure data quality?",
    # "What metrics are related to operational efficiency?",
    # "Tell me about renewable energy factors in our KPIs"

    test_queries = [
        # "What is Access Cost?",
        "Explain the Financial Perspective in BSC"
    ]

    run_test_queries(test_queries, k=3)





