#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaptive Retrieval Evaluation Harness
--------------------------------------
Use this as a module or script to evaluate your adaptive retrieval pipeline.

Main entrypoint for Python code:
    from evaluate_adaptive_retrieval import run_evaluation
    results = run_evaluation(["path/to/testset.json", ...])

It will print aggregate metrics and return a dictionary with results.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

# =========== Plug your retriever here ===========
def get_retriever():
    """
    Import and construct your AdaptiveRetriever here.
    Example:
        from RAG.Retrieval.adaptive_retriever import AdaptiveRetriever
        return AdaptiveRetriever(config=...)
    """
    from RAG.Retrieval.adaptive_retriever import AdaptiveRetriever

    persist_directory = "RAG/ProcessedDocuments/chroma_db"
    model_name = "all-MiniLM-L6-v2"
    logger.info(f"Loading vector store from {persist_directory}")
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embedding_function)

    return AdaptiveRetriever(
    vectorstore=vectorstore,
    k_init=50,
    pool_cap=20
    )
    raise RuntimeError("Implement get_retriever() to return your retriever instance.")


# =========== Utilities ===========
def load_testset(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Test set at {path} is not a list.")
    return data


def doc_to_id(doc: Any) -> str:
    try:
        meta = getattr(doc, "metadata", None)
        if isinstance(meta, dict):
            t = meta.get("type")
            n = meta.get("name")
            if t and n:
                return f"{t}::{n}"
    except Exception:
        pass
    if isinstance(doc, dict):
        t = doc.get("type")
        n = doc.get("name")
        if t and n:
            return f"{t}::{n}"
    return str(doc)


def precision_recall_f1(selected: List[str], gold: List[str]) -> Tuple[float, float, float]:
    sel, gol = set(selected), set(gold)
    tp = len(sel & gol)
    fp = len(sel - gol)
    fn = len(gol - sel)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def mrr(selected: List[str], gold: List[str]) -> float:
    gold_set = set(gold)
    for rank, doc_id in enumerate(selected, start=1):
        if doc_id in gold_set:
            return 1.0 / rank
    return 0.0


# =========== Core Evaluation ===========
def evaluate_queries(retriever, items: List[Dict[str, Any]], stage="final", fixed_k=None):
    results = []
    for ex in items:
        question = ex.get("question", "")
        gold = list(ex.get("ground_truth_docs", []))

        t0 = time.time()
        step1 = retriever.step1_wide_retrieval(question)
        t1 = time.time()
        step2 = retriever.step2_rerank(step1, question)
        t2 = time.time()
        step3 = retriever.step3_adaptive_selection(step2, question)
        t3 = time.time()

        # Choose stage
        if stage == "wide":
            stage_docs = step1
        elif stage == "reranked":
            stage_docs = step2
        else:
            stage_docs = step3

        selected_ids = [doc_to_id(d) for d in stage_docs]
        if fixed_k is not None:
            selected_ids = selected_ids[:fixed_k]

        prec, rec, f1 = precision_recall_f1(selected_ids, gold)
        rr = mrr(selected_ids, gold) if gold else 0.0

        results.append({
            "question": question,
            "gold": gold,
            "selected": selected_ids,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "mrr": rr,
            "t_total_ms": (t3 - t0) * 1000,
        })
    return results


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(results)
    if n == 0:
        return {}
    avg = lambda k: sum(r[k] for r in results) / n
    return {
        "n": n,
        "precision": avg("precision"),
        "recall": avg("recall"),
        "f1": avg("f1"),
        "mrr": avg("mrr"),
    }


# =========== Public API ===========
def run_evaluation(testset_paths: List[str], stage="final", fixed_k=None) -> Dict[str, Any]:
    items = []
    for p in testset_paths:
        items.extend(load_testset(Path(p)))

    retriever = get_retriever()
    results = evaluate_queries(retriever, items, stage=stage, fixed_k=fixed_k)
    agg = aggregate(results)

    print("\n===== Evaluation Results =====")
    print(f"N={agg['n']} | P={agg['precision']:.3f} | R={agg['recall']:.3f} | "
          f"F1={agg['f1']:.3f} | MRR={agg['mrr']:.3f}")

    return {"aggregate": agg, "per_query": results}
