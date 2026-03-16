import fitz
import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PageContent:
    text: str
    page_num: int
    source: str


def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def load_pdf(pdf_path: str) -> list[PageContent]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = clean_text(text)
        if len(text) < 100:
            continue
        pages.append(PageContent(
            text=text,
            page_num=page_num,
            source=path.name
        ))

    doc.close()
    print(f"Loaded {len(pages)} pages from {path.name}")
    return pages


def load_pdfs_from_folder(folder_path: str) -> list[PageContent]:
    folder = Path(folder_path)
    all_pages = []
    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDFs found in {folder_path}")

    print(f"Found {len(pdf_files)} PDFs...")
    for pdf_file in pdf_files:
        pages = load_pdf(str(pdf_file))
        all_pages.extend(pages)

    print(f"Total pages extracted: {len(all_pages)}")
    return all_pages