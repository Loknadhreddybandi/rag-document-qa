import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ingest.pdf_loader import load_pdf
from src.ingest.chunker import chunk_pages
from src.retrieval.vector_store import VectorStore
from src.generation.generator import RAGGenerator
from dotenv import load_dotenv

load_dotenv()

TOP_K = int(os.getenv("TOP_K", 3))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")

app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload PDFs, ask questions, get cited answers.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

vector_store = VectorStore()
generator = RAGGenerator()
loaded_docs: list[str] = []


@app.on_event("startup")
async def startup():
    if vector_store.load(FAISS_INDEX_PATH):
        print("Existing index loaded.")
    else:
        print("No index found. Upload a PDF to start.")


class AskRequest(BaseModel):
    question: str
    top_k: int = TOP_K


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_ms: int
    model: str
    chunks_used: int


class UploadResponse(BaseModel):
    filename: str
    pages_extracted: int
    chunks_created: int
    total_chunks_in_index: int
    message: str


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    total_chunks: int
    docs_loaded: list[str]


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "index_loaded": vector_store.index is not None,
        "total_chunks": vector_store.total_chunks,
        "docs_loaded": loaded_docs
    }


@app.get("/stats")
def stats():
    return {
        "total_chunks": vector_store.total_chunks,
        "docs_loaded": loaded_docs,
        "embedding_model": os.getenv("EMBED_MODEL"),
        "llm_model": os.getenv("GROQ_MODEL")
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        pages = load_pdf(tmp_path)
        for p in pages:
            p.source = file.filename

        chunks = chunk_pages(pages)

        if vector_store.index is None:
            vector_store.build(chunks)
        else:
            vector_store.add(chunks)

        vector_store.save(FAISS_INDEX_PATH)

        if file.filename not in loaded_docs:
            loaded_docs.append(file.filename)

        return UploadResponse(
            filename=file.filename,
            pages_extracted=len(pages),
            chunks_created=len(chunks),
            total_chunks_in_index=vector_store.total_chunks,
            message=f"Successfully ingested {file.filename}!"
        )
    finally:
        os.unlink(tmp_path)


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if vector_store.index is None or vector_store.total_chunks == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a PDF first.")

    retrieved = vector_store.search(request.question, top_k=request.top_k)

    if not retrieved:
        raise HTTPException(status_code=404, detail="No relevant content found.")

    result = generator.generate(request.question, retrieved)

    return AskResponse(**result)