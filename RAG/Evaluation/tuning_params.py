import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from RAG.Retrieval.retriever import setup_retriever


def load_test_set(path):
    """Load a JSON list of questions with ground_truth_docs."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_grid(test_questions, persist_directory, ks, thresholds):
    """
    Run grid-search over ks and raw-cosine thresholds.
    Returns a DataFrame with columns: k, threshold, precision, recall, f1,
    retrieval_mrr, avg_similarity, rank_of_first_match
    """
    rows = []
    n_q = len(test_questions)

    for k in ks:
        retriever = setup_retriever(persist_directory=persist_directory, k=k)

        for thr in thresholds:
            total_tp = 0  # total true positives across all queries
            total_ret = 0  # total retrieved (above threshold)
            q_with_hit = 0  # queries with at least one correct doc
            q_with_any = 0  # queries returning any doc above threshold
            total_gt = 0  # total ground-truth docs across queries (for true recall)

            reciprocal_ranks = []  # for MRR calculation
            all_similarities = []  # for avg_similarity
            first_match_ranks = []  # for rank_of_first_match

            for q in test_questions:
                # retrieve top-k
                docs_and_dist = retriever.vectorstore.similarity_search_with_score(
                    q["question"], k=k
                )

                # filter by threshold & deduplicate by document ID
                filtered_ids = []
                filtered_similarities = []
                seen = set()
                first_match_rank = None

                for rank, (doc, dist) in enumerate(docs_and_dist, 1):
                    cos_sim = 1.0 - dist
                    if cos_sim >= thr:
                        doc_id = f"{doc.metadata.get('type')}::{doc.metadata.get('name')}"
                        if doc_id not in seen:
                            seen.add(doc_id)
                            filtered_ids.append(doc_id)
                            filtered_similarities.append(cos_sim)

                            if first_match_rank is None and doc_id in q.get("ground_truth_docs", []):
                                first_match_rank = rank

                gt = set(q.get("ground_truth_docs", []))
                hits = len(gt.intersection(filtered_ids))

                total_tp += hits
                total_ret += len(filtered_ids)
                total_gt += len(gt)

                if hits > 0:
                    q_with_hit += 1
                if len(filtered_ids) > 0:
                    q_with_any += 1

                if first_match_rank is not None:
                    reciprocal_ranks.append(1.0 / first_match_rank)
                    first_match_ranks.append(first_match_rank)
                else:
                    reciprocal_ranks.append(0.0)
                    first_match_ranks.append(k + 1)

                all_similarities.extend(filtered_similarities)

            precision = total_tp / total_ret if total_ret > 0 else 0.0
            recall = total_tp / total_gt if total_gt > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall > 0) else 0.0
            false_positive_rate = q_with_any / n_q
            retrieval_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
            avg_similarity = sum(all_similarities) / len(all_similarities) if all_similarities else 0.0
            rank_of_first_match = sum(first_match_ranks) / len(first_match_ranks) if first_match_ranks else 0.0

            rows.append({
                "k": k,
                "threshold": thr,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive_rate": false_positive_rate,
                "retrieval_mrr": retrieval_mrr,
                "avg_similarity": avg_similarity,
                "rank_of_first_match": rank_of_first_match
            })

    return pd.DataFrame(rows)



def plot_precision_recall(df: pd.DataFrame, title: str, save_path: str = None):
    """
    Expects a DataFrame with columns: k, threshold, precision, recall
    Plots recall on the x-axis and precision on the y-axis, one line per k.
    """
    plt.figure(figsize=(8, 6))
    for k in sorted(df['k'].unique()):
        sub = df[df['k'] == k].sort_values('recall')
        plt.plot(sub['recall'], sub['precision'], marker='o', label=f'k={k}')
        for _, row in sub.iterrows():
            plt.annotate(f"{row['threshold']}", (row['recall'], row['precision']),
                         textcoords="offset points", xytext=(3,-3), fontsize=8)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="top-k")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved at: {save_path}")

    plt.show()
