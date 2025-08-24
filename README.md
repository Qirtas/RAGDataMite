# RAGForDatamite

Prototype for a Retrieval-Augmented Generation (RAG) feature in DATAMITE using LangChain. 

What Does This Project Do?

This project builds a basic pipeline that:
- Preprocesses structured CSV files (content for KB)
- Converts them into LangChain-compatible `Document` objects
- Generates vector embeddings for LangChain docs
- Stores those embeddings in a vector DB
- Can retrieve documents based on users question
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

This project supports running the RAG API with **two different LLM providers**:
- [Anthropic Claude](https://www.anthropic.com/)  
- [DeepSeek (via OpenRouter)](https://openrouter.ai/)  

You can switch between LLMs without changing any code by setting the `LLM_PROVIDER` environment variable (`claude` or `deepseek`).  
We provide ready-made scripts for **macOS** (`.sh`) and **Windows PowerShell** (`.ps1`).

---

### 1. Prerequisites

- **Python**: version **3.10** is required.  
- **Conda** (recommended for environment management).  

---

### 2. Create Environment

To set up the environment, follow these steps:

```bash
# Create conda env with Python 3.10
conda create -n datamite_env python=3.10

# Activate it
conda activate datamite_env

# Install dependencies
pip install -r requirements.txt
```

---

### 3. Environment variables

You need to export API keys depending on which LLM provider you want to use.

Claude (Anthropic):

```bash
# macOS
export ANTHROPIC_API_KEY=""

# Windows
$env:ANTHROPIC_API_KEY = ""
```

DeepSeek (via OpenRouter):

```bash
# macOS
export OPENROUTER_API_KEY=""

# Windows
$env:OPENROUTER_API_KEY = ""
```
---

### 4. Scripts

We provide helper scripts for both platforms.

macOS

```bash
# Run with Claude:

./run_claude.sh "What is CAPEX?"

# Run with DeepSeek:

./run_deepseek.sh "What is CAPEX?"
```

Windows

```bash
# Run with Claude:

.\run_claude.ps1 "What is CAPEX?"

# Run with DeepSeek:

.\run_deepseek.ps1 "What is CAPEX?"
```
---


