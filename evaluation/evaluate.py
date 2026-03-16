import requests
import json
import pandas as pd
from datetime import datetime

API_URL = "http://localhost:8000"

# These are 20 real questions based on your 3 AI papers
# Question + what a good answer should contain
TEST_QUESTIONS = [
    {
        "question": "What is the transformer architecture?",
        "expected_keywords": ["attention", "encoder", "decoder", "self-attention"]
    },
    {
        "question": "What is the attention mechanism?",
        "expected_keywords": ["query", "key", "value", "weights"]
    },
    {
        "question": "What training data was used for GPT-4?",
        "expected_keywords": ["training", "data", "tokens", "pretraining"]
    },
    {
        "question": "What are the limitations of large language models?",
        "expected_keywords": ["hallucination", "bias", "limitations", "errors"]
    },
    {
        "question": "How does multi-head attention work?",
        "expected_keywords": ["heads", "attention", "parallel", "concatenate"]
    },
    {
        "question": "What is LLaMA 2 trained on?",
        "expected_keywords": ["tokens", "training", "data", "pretraining"]
    },
    {
        "question": "What is RLHF?",
        "expected_keywords": ["reinforcement", "human", "feedback", "reward"]
    },
    {
        "question": "What is the context window size?",
        "expected_keywords": ["tokens", "context", "length", "window"]
    },
    {
        "question": "How is model performance evaluated?",
        "expected_keywords": ["benchmark", "evaluation", "accuracy", "score"]
    },
    {
        "question": "What is positional encoding?",
        "expected_keywords": ["position", "encoding", "sequence", "embedding"]
    },
    {
        "question": "What are the safety measures in GPT-4?",
        "expected_keywords": ["safety", "alignment", "harmful", "RLHF"]
    },
    {
        "question": "How does the feed forward network work in transformers?",
        "expected_keywords": ["feed", "forward", "linear", "activation"]
    },
    {
        "question": "What is fine tuning?",
        "expected_keywords": ["fine", "tuning", "pretrained", "task"]
    },
    {
        "question": "What is the difference between GPT-4 and previous models?",
        "expected_keywords": ["improved", "performance", "capabilities", "benchmark"]
    },
    {
        "question": "How are embeddings used in language models?",
        "expected_keywords": ["embedding", "vector", "representation", "tokens"]
    },
    {
        "question": "What is beam search?",
        "expected_keywords": ["beam", "search", "decoding", "sequence"]
    },
    {
        "question": "What is the role of layer normalization?",
        "expected_keywords": ["normalization", "layer", "training", "stable"]
    },
    {
        "question": "How does LLaMA handle long contexts?",
        "expected_keywords": ["context", "length", "attention", "tokens"]
    },
    {
        "question": "What datasets were used to evaluate the transformer?",
        "expected_keywords": ["dataset", "WMT", "evaluation", "translation"]
    },
    {
        "question": "What is temperature in language model generation?",
        "expected_keywords": ["temperature", "sampling", "probability", "generation"]
    }
]


def check_answer_quality(answer: str, expected_keywords: list[str]) -> float:
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(found / len(expected_keywords), 2)


def run_evaluation():
    print("=" * 60)
    print("RAG PIPELINE EVALUATION")
    print("=" * 60)

    # Check API is running
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        print(f"API Status: {health['status']}")
        print(f"Total chunks in index: {health['total_chunks']}")
        print(f"Documents loaded: {health['docs_loaded']}")
    except Exception as e:
        print(f"ERROR: API not reachable. Start FastAPI first.\n{e}")
        return

    if health["total_chunks"] == 0:
        print("ERROR: No documents indexed. Upload PDFs first.")
        return

    print(f"\nRunning {len(TEST_QUESTIONS)} test questions...\n")

    results = []
    total_latency = 0
    passed = 0

    for i, test in enumerate(TEST_QUESTIONS, 1):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": test["question"], "top_k": 3},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                quality_score = check_answer_quality(
                    data["answer"],
                    test["expected_keywords"]
                )
                total_latency += data["latency_ms"]

                if quality_score >= 0.5:
                    passed += 1
                    status = "PASS"
                else:
                    status = "FAIL"

                results.append({
                    "question": test["question"][:50] + "...",
                    "status": status,
                    "quality_score": quality_score,
                    "latency_ms": data["latency_ms"],
                    "chunks_used": data["chunks_used"],
                    "sources": [s["source"] for s in data["sources"]]
                })

                print(f"Q{i:02d} [{status}] score={quality_score:.2f} latency={data['latency_ms']}ms | {test['question'][:45]}...")

            else:
                print(f"Q{i:02d} [ERROR] {response.json().get('detail')}")
                results.append({
                    "question": test["question"][:50] + "...",
                    "status": "ERROR",
                    "quality_score": 0,
                    "latency_ms": 0,
                    "chunks_used": 0,
                    "sources": []
                })

        except Exception as e:
            print(f"Q{i:02d} [ERROR] {e}")

    # Summary
    avg_latency = total_latency // len(TEST_QUESTIONS)
    pass_rate = round(passed / len(TEST_QUESTIONS) * 100, 1)
    avg_quality = round(
        sum(r["quality_score"] for r in results) / len(results), 2
    )

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions : {len(TEST_QUESTIONS)}")
    print(f"Passed          : {passed}/{len(TEST_QUESTIONS)}")
    print(f"Pass rate       : {pass_rate}%")
    print(f"Avg quality     : {avg_quality:.2f}")
    print(f"Avg latency     : {avg_latency}ms")
    print("=" * 60)
    print("\nRESUME METRICS YOU CAN USE:")
    print(f"  - Evaluated RAG pipeline on {len(TEST_QUESTIONS)} questions")
    print(f"  - Achieved {pass_rate}% answer pass rate")
    print(f"  - Average response latency: {avg_latency}ms")
    print(f"  - Average answer quality score: {avg_quality}")

    # Save results to CSV
    df = pd.DataFrame(results)
    filename = f"evaluation/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"\nDetailed results saved to: {filename}")


if __name__ == "__main__":
    run_evaluation()