# RAG Document Q&A System

End-to-end Retrieval-Augmented Generation pipeline that lets users upload any PDF and ask questions in natural language — returning accurate, cited answers with exact page references.

---

## Metrics

| Metric | Value |
|--------|-------|
| Answer success rate | **85%** across 20 test queries |
| Cosine similarity retrieval score | **0.544** |
| Minimum retrieval latency | **126ms** |
| Avg end-to-end latency | **~5s** (includes Groq free-tier API) |
| Chunks indexed | **688 chunks across 189 pages** |
| Documents tested | **3 AI research papers** |

---

## Architecture

```
PDF Upload (up to 200 pages)
        ↓
pdf_loader.py — PyMuPDF text extraction, clean per page
        ↓
chunker.py — LangChain RecursiveCharacterTextSplitter
             1024-token chunks, 50-token overlap
        ↓
vector_store.py — sentence-transformers (all-MiniLM-L6-v2)
                  384-dim embeddings → FAISS IndexFlatIP
                  persisted to disk (index.faiss + metadata.json)
        ↓
User Question
        ↓
vector_store.py — embed query → cosine similarity search → top-5 chunks
        ↓
generator.py — prompt construction (system + context + question)
               LLaMA-3 8B via Groq API → cited answer
        ↓
main.py (FastAPI REST API) + app.py (Streamlit UI)
```

---

## Stack

Python · sentence-transformers · FAISS · LangChain · LLaMA-3 · Groq API · FastAPI · Streamlit · PyMuPDF · Docker

---

## Run

```bash
git clone https://github.com/Loknadhreddybandi/rag-document-qa
cd rag-document-qa
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt && pip install tf-keras

# Add your GROQ_API_KEY to .env
cp .env.example .env

# Terminal 1 — backend
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — frontend
streamlit run ui/app.py

# Ingest all PDFs at once
python ingest_all.py
```

---

## API Endpoints

- `GET /health` — health check, index status, loaded docs
- `GET /stats` — chunk count, embedding model, LLM model
- `POST /upload` — upload PDF, extract, chunk, embed, index
- `POST /ask` — ask question, returns answer + sources + latency

---

## Dataset / Documents Used

3 AI research papers used for evaluation:
- **Attention Is All You Need** — Vaswani et al., 2017 (arXiv:1706.03762)
- **GPT-4 Technical Report** — OpenAI, 2023 (arXiv:2303.08774)
- **LLaMA 2** — Touvron et al., Meta AI, 2023 (arXiv:2307.09288)

Labels derived from a custom 20-question evaluation set with keyword-based quality scoring.

---

## Author

**Bandi Venkata Loknadh Reddy**
MS Data Science & AI — University of Central Missouri
[GitHub](https://github.com/Loknadhreddybandi) · [LinkedIn](https://linkedin.com/in/bandivenkataloknadhreddy)
