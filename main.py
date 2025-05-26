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

if __name__ == '__main__':

    # 1. Preprocess CSV Files to clean for JSON related issues


    output_dir = "RAG/Content/ProcessedFiles"

    try:
        processed_files = preprocess_data(output_dir)

        print("\nPreprocessing Summary:")
        print("=====================")
        for file_type, file_path in processed_files.items():
            print(f"{file_type}: {file_path}")

        print("\nTo use these files with KB.py:")
        print("```python")
        print("from KB import load_csv_as_documents, load_all_csvs_as_documents")
        print("")
        print("# Define processed file paths")
        print("csv_files = {")
        for file_type, file_path in processed_files.items():
            print(f"    '{file_type}': '{file_path}',")
        print("}")
        print("")
        print("# Load all documents")
        print("all_docs = load_all_csvs_as_documents(csv_files)")
        print("print(f\"Loaded {len(all_docs)} total documents\")")
        print("```")

    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        sys.exit(1)
    


    # 2. Documents Ingestion - converting to Langchain documents


    documents = ingest_documents()

    if documents:
        print("\nSample document:")
        print("-" * 80)

        sample_doc = next((doc for doc in documents if doc.metadata.get("type") == "KPI"), documents[0])

        print("Content:")
        print(sample_doc.page_content[:500] + "..." if len(sample_doc.page_content) > 500 else sample_doc.page_content)
        print("-" * 80)
        print("Metadata:")
        print(sample_doc.metadata)
    


    # 3. Analysing if there is a need for text chunking

    with open('RAG/ProcessedDocuments/all_documents.pkl', 'rb') as f:
        documents = pickle.load(f)

    doc_lengths = defaultdict(list)
    for doc in documents:
        doc_type = doc.metadata.get('type', 'Unknown')
        length = len(doc.page_content)
        doc_lengths[doc_type].append(length)

    for doc_type, lengths in doc_lengths.items():
        print(f"\n{doc_type} Documents:")
        print(f"  Count: {len(lengths)}")
        print(f"  Average length: {np.mean(lengths):.1f} characters")
        print(f"  Min length: {min(lengths)} characters")
        print(f"  Max length: {max(lengths)} characters")

        if lengths:
            longest_idx = np.argmax(lengths)
            longest_doc = next((doc for i, doc in enumerate(documents)
                                if doc.metadata.get('type') == doc_type
                                and i == longest_idx), None)
            if longest_doc:
                print(f"  Longest document name: {longest_doc.metadata.get('name', 'Unknown')}")
                print(f"  First 200 chars: {longest_doc.page_content[:200]}...")



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





