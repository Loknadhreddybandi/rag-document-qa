import requests

API_URL = "http://127.0.0.1:8000"

pdfs = [
    ("data/pdfs/attention.pdf", "attention.pdf"),
    ("data/pdfs/gpt4.pdf", "gpt4.pdf"),
    ("data/pdfs/llama2.pdf", "llama2.pdf")
]

for path, name in pdfs:
    print(f"Ingesting {name}...")
    with open(path, "rb") as f:
        r = requests.post(
            f"{API_URL}/upload",
            files={"file": (name, f, "application/pdf")},
            timeout=120
        )
        data = r.json()
        print(f"  Pages: {data['pages_extracted']}")
        print(f"  Chunks: {data['chunks_created']}")
        print(f"  Total index: {data['total_chunks_in_index']}")
        print()

print("All PDFs ingested!")
