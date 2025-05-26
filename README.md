# RAGForDatamite

Prototype for a Retrieval-Augmented Generation (RAG) feature in DATAMITE using LangChain. 

What Does This Project Do?

This project builds a basic pipeline that:
- Preprocesses structured CSV files (content for KB)
- Converts them into LangChain-compatible `Document` objects
- Generates vector embeddings for LangChain docs
- Stores those embeddings in a vector DB


## Project Structure

RAGForDatamite/
├── RAG -> Main code folder
├── Content/ files related to KB like raw csv files
├── KB/ # Preprocessing, ingestion, embeddings, vector store
├── Retrieval/ # Querying logic and evaluation
├── ProcessedDocuments/ # Pickled LangChain documents
├── main.py # Main pipeline for preprocessing, embedding & retrieval

## How to Run
Basic pipeline to test retrieval from existing documents:

python main.py

### Example Retrieval Queries
Inside main.py, you can modify the list of test queries:

test_queries = [
    "Explain the Financial Perspective in BSC",
    "What is Access Cost?"
]

## Running Tests

python -m unittest discover tests

## Next Steps

Test with other embeddings models and vector DBs

Analyze similarity scores by retrieval for validation

Splitting complex questions into multiple sub-questions

Connecting retrieval to LLM

Adding validation search
