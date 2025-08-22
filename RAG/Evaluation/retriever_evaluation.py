import json

import matplotlib.pyplot as plt
import pandas as pd

from RAG.Retrieval.retriever import setup_retriever


def load_questions(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def collect_scores(questions: list, retriever, k: int = 1) -> pd.DataFrame:
    """
    For each question, run similarity_search_with_score and record:
      - distance (d)
      - rescaled sim_score = 1/(1+d)
      - raw_cosine_sim = 1 - d
    """
    records = []
    for q in questions:
        results = retriever.vectorstore.similarity_search_with_score(q["question"], k=k)
        if not results:
            # no hits, skip
            continue
        doc, dist = results[0]
        sim_scaled = 1 / (1 + dist)
        cos_sim    = 1 - dist
        records.append({
            "question_id":       q["question_id"],
            "category":          q["category"],
            "distance":          dist,
            "sim_scaled":        sim_scaled,
            "cosine_similarity": cos_sim
        })
    return pd.DataFrame(records)

def summarize_and_plot(df: pd.DataFrame, k=0):
    summary = (
        df
        .groupby("category")[["distance", "sim_scaled", "cosine_similarity"]]
        .agg(["count", "min", "median", "mean", "max"])
    )
    print("\n=== Similarity Score Summary by Category ===")
    print(summary)

    with open("RAG/Evaluation/data/TestSet/similarity_summary.txt", "w") as f:
        f.write("=== Similarity Score Summary by Category ===\n")
        f.write(summary.to_string())

    cats = df["category"].unique()
    n = len(cats)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, cat in zip(axes, cats):
        data = df[df["category"] == cat]["cosine_similarity"]
        ax.hist(data, bins=20, edgecolor='black')
        if cat == "direct_domain_relevant_relationship":
            cat = "Domain Relevant 2 ground truths"
        ax.set_title(f"{cat}")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Frequency")
    fig.suptitle("Distribution of Raw Cosine Similarity by Category")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
