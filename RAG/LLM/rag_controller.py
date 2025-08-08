from RAG.LLM.simple_claude_rag import ask_claude_with_docs
from RAG.LLM.validation_search import validation_search
from RAG.Retrieval.retriever import get_retrieval_results


def rag_with_validation(query, min_similarity=0.40,
                                 persist_directory="RAG/ProcessedDocuments/chroma_db", k=3, retriever=None):
    """
    RAG system with semantic validation step
    """
    print(f"\n{'=' * 80}")
    print(f"Query: {query}")
    print(f"{'=' * 80}")

    # Step 1: Initial retrieval from KB and quality control
    results = get_retrieval_results(query, persist_directory=persist_directory, k=k)

    if not results or results[0][1] < min_similarity:
        best_similarity = results[0][1] if results else 0
        print(f"Best similarity ({best_similarity:.4f}) below threshold ({min_similarity})")
        return {
            "answer": "I don't have relevant information to answer this question.",
            "sources": [],
            "quality_check": "failed",
            "validation": None
        }

    # Step 2: Filter documents and generate answer using claude
    filtered_docs = [result[0] for result in results if result[1] >= min_similarity]
    print(f"Found {len(filtered_docs)} documents above similarity threshold ({min_similarity})")

    # Sending Query + retrieved docs to LLM
    claude_result = ask_claude_with_docs(query, filtered_docs)
    print(f"Claude Result Before Validation Search: {claude_result}")

    # Step 3: Perform validation search
    validation_result = validation_search(
        original_question=query,
        llm_answer=claude_result["answer"],
        original_docs=filtered_docs,
        persist_directory=persist_directory
    )

    # Step 4: Combine results
    final_result = {
        "answer": claude_result["answer"],
        "sources": claude_result["sources"],
        "quality_check": "passed",
        "validation": validation_result,
        "confidence": validation_result["confidence"],
        "validation_status": validation_result["validation_status"],
        "semantic_scores": validation_result["detailed_scores"]
    }

    return final_result