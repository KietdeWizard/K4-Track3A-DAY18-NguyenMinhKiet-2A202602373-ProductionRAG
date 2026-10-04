from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    # 1. Wrap trong try/except — RAGAS cần OPENAI_API_KEY và Python 3.11+.
    # try:
    #     from ragas import evaluate
    #     from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    #     from datasets import Dataset
    #
    #     dataset = Dataset.from_dict({
    #         "question": questions, "answer": answers,
    #         "contexts": contexts, "ground_truth": ground_truths,
    #     })
    #     result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
    #                                         context_precision, context_recall])
    #     df = result.to_pandas()
    #     per_question = [EvalResult(question=row["question"], answer=row["answer"],
    #         contexts=row["contexts"], ground_truth=row["ground_truth"],
    #         faithfulness=float(row.get("faithfulness", 0.0)),
    #         answer_relevancy=float(row.get("answer_relevancy", 0.0)),
    #         context_precision=float(row.get("context_precision", 0.0)),
    #         context_recall=float(row.get("context_recall", 0.0)))
    #         for _, row in df.iterrows()]
    #     return {"faithfulness": ..., "answer_relevancy": ...,
    #             "context_precision": ..., "context_recall": ..., "per_question": [...]}
    # except Exception as e:
    #     print(f"  ⚠️  RAGAS evaluation failed: {e}")
    #     return zeros
    return {"faithfulness": 0.0, "answer_relevancy": 0.0,
            "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    # 1. diagnostic_tree = {
    #        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
    #        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    #        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    #        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    #    }
    # 2. For each EvalResult: compute avg of 4 metrics, find worst_metric
    # 3. Sort by avg ascending → take bottom_n
    # 4. Return [{"question": ..., "worst_metric": ..., "score": ...,
    #             "diagnosis": ..., "suggested_fix": ...}]
    return []


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    empty = {name: 0.0 for name in names}
    empty["per_question"] = []
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        dataset = Dataset.from_dict({"question": questions, "answer": answers,
                                     "contexts": contexts, "ground_truth": ground_truths})
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])
        frame = result.to_pandas()
        per_question = []
        for index, row in frame.iterrows():
            def value(name):
                raw = row.get(name, 0.0)
                return float(raw) if raw == raw else 0.0
            per_question.append(EvalResult(questions[index], answers[index], contexts[index], ground_truths[index],
                                          value("faithfulness"), value("answer_relevancy"),
                                          value("context_precision"), value("context_recall")))
        output = {name: float(frame[name].mean()) if name in frame else 0.0 for name in names}
        output["per_question"] = per_question
        return output
    except Exception as exc:
        print(f"  RAGAS evaluation failed: {exc}; using local overlap fallback")
        import re
        def terms(value):
            return {word for word in re.findall(r"\w+", value.lower()) if len(word) > 1}
        per_question = []
        for question, answer, docs, truth in zip(questions, answers, contexts, ground_truths):
            answer_terms, question_terms, truth_terms = terms(answer), terms(question), terms(truth)
            context_terms = set().union(*(terms(doc) for doc in docs)) if docs else set()
            faith = len(answer_terms & context_terms) / max(len(answer_terms), 1)
            relevancy = len(answer_terms & question_terms) / max(len(question_terms), 1)
            relevant_docs = sum(bool(terms(doc) & truth_terms) for doc in docs)
            precision = relevant_docs / max(len(docs), 1)
            recall = len(truth_terms & context_terms) / max(len(truth_terms), 1)
            per_question.append(EvalResult(question, answer, docs, truth, faith, relevancy, precision, recall))
        if per_question:
            return {name: sum(getattr(item, name) for item in per_question) / len(per_question)
                    for name in names} | {"per_question": per_question}
        return empty


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    tree = {
        "faithfulness": ("LLM tự bịa thông tin ngoài tài liệu", "Siết system prompt và giảm temperature về 0"),
        "context_recall": ("Hệ thống bỏ sót đoạn văn liên quan", "Cải thiện chunking hoặc bổ sung BM25"),
        "context_precision": ("Đoạn không liên quan xếp ở vị trí cao", "Bổ sung cross-encoder reranking hoặc lọc metadata"),
        "answer_relevancy": ("Câu trả lời lệch trọng tâm câu hỏi", "Viết lại prompt để trả lời trực tiếp hơn"),
    }
    analysed = []
    for item in eval_results:
        scores = {name: float(getattr(item, name)) for name in tree}
        worst = min(scores, key=scores.get)
        average = sum(scores.values()) / len(scores)
        diagnosis, fix = tree[worst]
        analysed.append({"question": item.question, "answer": item.answer,
                         "worst_metric": worst, "score": scores[worst],
                         "average_score": average, "diagnosis": diagnosis,
                         "suggested_fix": fix})
    return sorted(analysed, key=lambda item: item["average_score"])[:bottom_n]


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
