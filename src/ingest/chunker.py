import os
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.ingest.pdf_loader import PageContent
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))


@dataclass
class Chunk:
    text: str
    source: str
    page_num: int
    chunk_id: str


def chunk_pages(pages: list[PageContent]) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    chunks = []
    for page in pages:
        texts = splitter.split_text(page.text)
        for idx, text in enumerate(texts):
            if len(text.strip()) < 30:
                continue
            chunks.append(Chunk(
                text=text.strip(),
                source=page.source,
                page_num=page.page_num,
                chunk_id=f"{page.source}_p{page.page_num}_c{idx}"
            ))

    print(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks