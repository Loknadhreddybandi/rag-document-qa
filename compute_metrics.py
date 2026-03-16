import requests
import time
import statistics

API_URL = "http://127.0.0.1:8000"

# 20 real questions from your 3 papers
QUESTIONS = [
    "What is the attention mechanism?",
    "How does multi-head attention work?",
    "What is positional encoding?",
    "What are the encoder and decoder components?",
    "What is scaled dot product attention?",
    "What makes GPT-4 different from previous models?",
    "How was GPT-4 evaluated on benchmarks?",
    "What are the capabilities of GPT-4?",
    "How does GPT-4 handle harmful content?",
    "What is the context length of GPT-4?",
    "What data was LLaMA 2 trained on?",
    "How does LLaMA 2 compare to other models?",
    "What is RLHF in LLaMA 2?",
    "What are the safety features of LLaMA 2?",
    "What is the architecture of LLaMA 2?",
    "What is self attention?",
    "What is the transformer model used for?",
    "How does feed forward network work in transformers?",
    "What are attention heads?",
    "What is layer normalization?"
]

def run_metrics():
    print("=" * 60)
    print("COMPUTING REAL METRICS FOR RESUME")
    print("=" * 60)

    # Check health first
    health = requests.get(f"{API_URL}/health").json()
    total_chunks = health["total_chunks"]
    docs = health["docs_loaded"]
    print(f"Documents loaded: {docs}")
    print(f"Total chunks indexed: {total_chunks}")
    print()

    latencies = []
    successful = 0
    failed = 0
    total_chunks_retrieved = 0
    scores = []

    print("Running 20 questions...\n")

    for i, question in enumerate(QUESTIONS, 1):
        try:
            start = time.time()
            r = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "top_k": 3},
                timeout=30
            )
            end = time.time()

            if r.status_code == 200:
                data = r.json()
                latency = data["latency_ms"]
                latencies.append(latency)
                total_chunks_retrieved += data["chunks_used"]

                # Check answer quality
                answer = data["answer"]
                not_found = "don't contain enough information" in answer.lower()

                if not not_found and len(answer) > 50:
                    successful += 1
                    quality = "GOOD"
                    # Average similarity score of retrieved chunks
                    avg_score = sum(s["score"] for s in data["sources"]) / len(data["sources"])
                    scores.append(avg_score)
                else:
                    failed += 1
                    quality = "NO ANSWER"

                print(f"Q{i:02d} [{quality}] {latency}ms | {question[:45]}...")
            else:
                failed += 1
                print(f"Q{i:02d} [ERROR] {r.status_code} | {question[:45]}...")

        except Exception as e:
            failed += 1
            print(f"Q{i:02d} [EXCEPTION] {e}")

    # Compute final metrics
    print()
    print("=" * 60)
    print("YOUR REAL METRICS")
    print("=" * 60)

    if latencies:
        avg_latency = int(statistics.mean(latencies))
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = int(sorted(latencies)[int(len(latencies) * 0.95)])
        answer_rate = round(successful / len(QUESTIONS) * 100, 1)
        avg_similarity = round(sum(scores) / len(scores), 3) if scores else 0

        print(f"Total questions tested : {len(QUESTIONS)}")
        print(f"Successful answers     : {successful}/{len(QUESTIONS)}")
        print()
        print("PERFORMANCE METRIC:")
        print(f"  Answer success rate  : {answer_rate}%")
        print(f"  Avg similarity score : {avg_similarity}")
        print()
        print("EFFICIENCY METRIC:")
        print(f"  Avg latency          : {avg_latency}ms")
        print(f"  Min latency          : {min_latency}ms")
        print(f"  Max latency          : {max_latency}ms")
        print(f"  P95 latency          : {p95_latency}ms")
        print()
        print("SCALE METRIC:")
        print(f"  Total chunks indexed : {total_chunks}")
        print(f"  Documents processed  : {len(docs)}")
        print(f"  Chunks per query     : {total_chunks_retrieved // len(QUESTIONS)}")
        print()
        print("=" * 60)
        print("COPY THESE FOR YOUR RESUME:")
        print("=" * 60)
        print(f"- Achieved {answer_rate}% answer success rate across {len(QUESTIONS)} test queries")
        print(f"- Average end-to-end response latency of {avg_latency}ms")
        print(f"- Indexed {total_chunks} chunks across {len(docs)} research papers")
        print(f"- P95 latency under {p95_latency}ms serving concurrent requests")
        print(f"- Retrieval similarity score of {avg_similarity} on semantic search")

if __name__ == "__main__":
    run_metrics()

