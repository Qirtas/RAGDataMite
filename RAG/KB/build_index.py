# build_index.py
import os
import logging
from typing import Optional
from RAG.KB.preprocess_data import preprocess_data
from RAG.KB.vector_DB import create_vectorstore
from RAG.KB.ingest_documents import ingest_documents
import sys
import pickle
import numpy as np
from collections import defaultdict
from RAG.KB.generating_embeddings import generate_embeddings

REQUIRED_CHROMA_FILES = {"chroma.sqlite3", "index"}
# Note: adjust if your Chroma layout differs; some installs create multiple shard dirs/files.

def _vectorstore_exists(persist_dir: str) -> bool:
    if not os.path.isdir(persist_dir):
        return False
    entries = set(os.listdir(persist_dir))
    # Heuristic: if we see any of the canonical Chroma files/dirs, assume it's built
    return any(name in entries for name in REQUIRED_CHROMA_FILES) or len(entries) > 0

def ensure_vectorstore(persist_dir: str, logger: Optional[logging.Logger] = None):
    """
    Build the vector store only if it's missing.
    """
    lg = logger or logging.getLogger("datamite")
    if _vectorstore_exists(persist_dir):
        lg.info(f"Vector store already present at '{persist_dir}'. Skipping index build.")
        return

    lg.info(f"No vector store found at '{persist_dir}'. Building index (steps 1–5)...")

    # 1. Preprocess CSV Files to clean for JSON related issues

    output_dir = "RAGDataMite/RAG/Content/ProcessedFiles"

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
        logger.info(
            sample_doc.page_content[:500] + "..." if len(sample_doc.page_content) > 500 else sample_doc.page_content)
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

    vectorstore = create_vectorstore()
    return vectorstore
