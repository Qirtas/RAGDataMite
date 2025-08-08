# RAGForDatamite

Prototype for a Retrieval-Augmented Generation (RAG) feature in DATAMITE using LangChain. 

What Does This Project Do?

This project builds a basic pipeline that:
- Preprocesses structured CSV files (content for KB)
- Converts them into LangChain-compatible `Document` objects
- Generates vector embeddings for LangChain docs
- Stores those embeddings in a vector DB
- Can retrieve documents based on users questions
- Can provide LLM (Claude) generated answers for users questions
- Performed optimisation for selecting k (number of documents to retrieve and similarity thresholds)


## Project Structure

RAGForDatamite/
├── RAG -> Main code folder
├── Content/ files related to KB like raw csv files
├── Evaluation/ test sets, tuning for k and similarity thresholds
├── KB/ # Preprocessing, ingestion, embeddings, vector store
├── LLM/ # Claude API integration, generating answers with LLM
├── Retrieval/ # Querying logic and evaluation
├── ProcessedDocuments/ # Pickled LangChain documents
├── main.py # Main pipeline for preprocessing, embedding & retrieval

## How to Run
Basic pipeline to test retrieval from existing documents:

python main.py

All components of RAG system have been called from main.py where I have added brief comment with each piece. You may uncomment relevant part and use it.

### Example Retrieval Queries
Inside main.py, you can modify the list of test queries:

test_queries = [
    "Explain the Financial Perspective in BSC",
    "What is Access Cost?"
]

## Running Tests

python -m unittest discover tests

