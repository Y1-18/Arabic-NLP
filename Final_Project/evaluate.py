"""
evaluate.py
-----------
Simple evaluation for the RAG system using Jina embedding similarity.
"""

import numpy as np

from rag import MultimodalRAG


class SimpleEvaluator:
    """Reuses the RAG's Jina embedder for similarity-based metrics."""

    def __init__(self, rag):
        self.rag = rag

    def _embed(self, text: str):
        out = self.rag.embedder.encode_text(
            texts=[text], task="retrieval", prompt_name="query",
        )
        return self.rag._to_numpy(out)[0]

    def _cosine(self, a: str, b: str) -> float:
        a_v = self._embed(a)
        b_v = self._embed(b)
        return float(np.dot(a_v, b_v) / (np.linalg.norm(a_v) * np.linalg.norm(b_v) + 1e-9))

    def faithfulness(self, answer, contexts):
        if not contexts or not answer:
            return 0.0
        return max(self._cosine(answer, c) for c in contexts)

    def answer_relevancy(self, query, answer):
        if not answer:
            return 0.0
        return self._cosine(query, answer)

    def context_precision(self, query, contexts):
        if not contexts:
            return 0.0
        return float(np.mean([self._cosine(query, c) for c in contexts]))


TEST_QUESTIONS = [
    {"query": "ما هو الموضوع الرئيسي للكتاب؟"},
    {"query": "من هو ابن فارس؟"},
    {"query": "ما عدد المضارب التي أوردها ابن فارس؟"},
]


def main():
    rag = MultimodalRAG()
    evaluator = SimpleEvaluator(rag)

    print("\n" + "=" * 70)
    print("RAG EVALUATION")
    print("=" * 70)

    all_scores = []
    for q in TEST_QUESTIONS:
        result = rag.answer(q["query"], k=3)
        scores = {
            "faithfulness": evaluator.faithfulness(result["answer"], result["contexts"]),
            "answer_relevancy": evaluator.answer_relevancy(q["query"], result["answer"]),
            "context_precision": evaluator.context_precision(q["query"], result["contexts"]),
        }
        all_scores.append(scores)

        print(f"\nQ: {q['query']}")
        print(f"A: {result['answer']}")
        print(f"  faithfulness     : {scores['faithfulness']:.3f}")
        print(f"  answer_relevancy : {scores['answer_relevancy']:.3f}")
        print(f"  context_precision: {scores['context_precision']:.3f}")

    print("\n" + "-" * 70)
    print("AVERAGES")
    print("-" * 70)
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        avg = np.mean([s[metric] for s in all_scores])
        print(f"  {metric:20s}: {avg:.3f}")


if __name__ == "__main__":
    main()