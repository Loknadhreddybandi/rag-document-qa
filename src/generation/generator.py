import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

SYSTEM_PROMPT = """You are a precise document assistant. Answer questions using ONLY the context provided below.

Rules:
- Answer directly and concisely based on the context.
- If the context does not contain the answer, say "The provided documents don't contain enough information to answer this question."
- Do not make up information. Do not use knowledge outside the context.
- Always cite the source document and page number at the end of your answer.
- Format citations as: [Source: filename, Page number]"""


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Chunk {i} | Source: {chunk['source']} | Page {chunk['page_num']}]\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return f"""CONTEXT FROM DOCUMENTS:
{context}

---

QUESTION: {question}

ANSWER (based only on the context above):"""


class RAGGenerator:
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env file")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL

    def generate(self, question: str, retrieved_chunks: list[dict]) -> dict:
        prompt = build_prompt(question, retrieved_chunks)

        start = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=512,
            top_p=0.9
        )

        latency_ms = int((time.time() - start) * 1000)
        answer = response.choices[0].message.content.strip()

        sources = []
        seen = set()
        for chunk in retrieved_chunks:
            key = (chunk["source"], chunk["page_num"])
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": chunk["source"],
                    "page_num": chunk["page_num"],
                    "score": round(chunk["score"], 4),
                    "preview": chunk["text"][:200] + "..."
                })

        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": latency_ms,
            "model": self.model,
            "chunks_used": len(retrieved_chunks)
        }