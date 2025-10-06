# RAGForDatamite

Prototype for a Retrieval-Augmented Generation (RAG) feature in DATAMITE using LangChain. 

What Does This Project Do?

This project provides a minimal working pipeline for running a Retrieval-Augmented Generation (RAG) workflow on the Datamite knowledge base.

It:

Builds or loads a vector database (Chroma) from your processed CSV content.

Retrieves the most relevant documents using fixed parameters (k and similarity_threshold).

Passes the retrieved context to a Large Language Model (LLM) such as Claude or DeepSeek.

### Prerequisites

- **Python**: version **3.10** is required.  
- **Conda** (For environment management).  

---
### 1. Get the code

Open a terminal (PowerShell on Windows, Terminal on macOS) and run:

```bash
git clone --recursive https://github.com/eduardovyhmeister/Datamite.git
cd Datamite
git submodule update --init --recursive --remote
```

Open the submodule folder in your editor:
Datamite/RAGDataMite

If using PyCharm, open RAGDataMite directly as the project.

---

### 2. Create a Python environment

```bash
conda create -n datamite_env python=3.10 -y
conda activate datamite_env
```
---

### 3. Install required packages

From inside RAGDataMite:
```bash
pip install -r requirements.txt
```
---

### 4. Configuration (config.yaml)

You can control everything from this single file:

```bash
# --- RAG settings ---
persist_dir: "RAG/ProcessedDocuments/chroma_db"

retrieval:
  k: 3
  similarity_threshold: 0.30  # cosine similarity threshold

# --- LLM settings ---
llm:
  provider: "deepseek"       # active provider: "claude" or "deepseek"

  claude:
    model: "claude-3-5-sonnet-20241022"
    api_key: "sk-ant-XXX"

  deepseek:
    model: "deepseek/deepseek-chat-v3-0324:free"
    api_key: "sk-or-XXX"
    base_url: "https://openrouter.ai/api/v1"
```

### 5. How to Switch Providers

Just change one line:

```bash
provider: "claude" or provider: "deepseek"
```

### 6. How to Run?
After setting up all the project in IDE, simply run the main.py or use terminal to ask a question:

```bash
python main.py --question "What is CAPEX?"
```

### 6. How to Change the Question

In main.py, look for this section:

```bash
result1 = rag_with_validation(
    "What is CAPEX?",
    min_similarity=similarity_threshold,
    k=k,
    llm=llm_client
)
```

You can change "What is CAPEX?" to any other question you want to test.

