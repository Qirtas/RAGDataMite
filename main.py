import argparse
import logging
import os

import sys
import yaml
from RAG.KB.build_index import ensure_vectorstore

from RAG.LLM.llm_provider import build_llm
from RAG.LLM.rag_controller import rag_with_validation

def load_config(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    print("Interpreter:", sys.executable)

    """
        1. Checking if there is a need to create vector DB or a DB already exists and there is no need.
    """

    parser = argparse.ArgumentParser(description="RAGForDatamite (minimal fixed-k pipeline)")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--question", "-q", required=False, help="Question to ask.")
    parser.add_argument("--mode", choices=["index", "ask"], default="ask",
                        help="index = build vector DB if missing; ask = ensure index, then answer question.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("datamite")

    cfg = load_config(args.config)
    persist_dir = cfg.get("persist_dir", "RAG/ProcessedDocuments/chroma_db")
    k = int(cfg.get("retrieval", {}).get("k", 3))
    similarity_threshold = float(cfg.get("retrieval", {}).get("similarity_threshold", 0.48))

    if args.mode == "index":
        ensure_vectorstore(persist_dir, logger=logger)
        logger.info("Index check/build finished.")
        return

    # mode == ask
    ensure_vectorstore(persist_dir, logger=logger)

    """
        2. Asking question to run the full RAG Pipeline including LLM
    """

    # Step 1: Build LLM client (based on config)

    llm_cfg = cfg.get("llm", {})
    provider = llm_cfg.get("provider", "claude").lower()

    # Pick correct section (claude / deepseek), you may change config in the config.yaml file
    prov_cfg = llm_cfg.get(provider, {})

    # Prefer env vars if they exist
    api_key = (
            os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or prov_cfg.get("api_key")
    )

    model = prov_cfg.get("model")
    base_url = prov_cfg.get("base_url")

    # Build the LLM client
    llm_client = build_llm(provider=provider, model=model, api_key=api_key, base_url=base_url)

    # Step 2: Load retrieval parameters from config ---
    retrieval_cfg = cfg.get("retrieval", {})
    k = int(retrieval_cfg.get("k", 3))
    similarity_threshold = float(retrieval_cfg.get("similarity_threshold", 0.30))

    # Step 3: Ask question
    result1 = rag_with_validation(
        "What is CAPEX?",
        min_similarity=similarity_threshold,
        k=k,
        llm=llm_client
    )
    print(f"Answer: {result1['answer']}")
    print(f"Number of sources: {len(result1['sources'])}")
    print()

if __name__ == "__main__":
    main()