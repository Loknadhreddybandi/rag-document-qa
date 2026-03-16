import streamlit as st
import requests
import os

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
    .source-card {
        background: #f8f9fa;
        border-left: 3px solid #7F77DD;
        border-radius: 4px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13px;
    }
    .score-badge {
        background: #E1F5EE;
        color: #085041;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📄 RAG Document Q&A")
    st.caption("Powered by LLaMA-3 + FAISS + sentence-transformers")
    st.divider()

    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF — research papers, reports, textbooks"
    )

    if uploaded_file:
        if st.button("Ingest PDF", type="primary", use_container_width=True):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    response = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                        timeout=120
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success("Ingested successfully!")
                        st.metric("Pages extracted", data["pages_extracted"])
                        st.metric("Chunks created", data["chunks_created"])
                        st.metric("Total index size", data["total_chunks_in_index"])
                        st.session_state["doc_loaded"] = True
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Upload failed')}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    st.divider()

    try:
        stats = requests.get(f"{API_URL}/stats", timeout=5).json()
        st.subheader("Index Stats")
        st.metric("Total chunks", stats.get("total_chunks", 0))
        docs = stats.get("docs_loaded", [])
        if docs:
            st.caption("Documents loaded:")
            for doc in docs:
                st.code(doc, language=None)
        st.caption(f"Embed: `{stats.get('embedding_model', 'N/A')}`")
        st.caption(f"LLM: `{stats.get('llm_model', 'N/A')}`")
    except:
        st.warning("API not reachable. Start FastAPI server first.")

    st.divider()
    st.caption("Built by Bandi Venkata Loknadh Reddy")
    st.caption("UCM — MS Data Science & AI, 2026")


st.title("Ask Your Documents")
st.caption("Upload a PDF in the sidebar, then ask any question about it.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])} chunks retrieved)"):
                for src in msg["sources"]:
                    st.markdown(f"""
<div class="source-card">
<strong>{src['source']}</strong> — Page {src['page_num']}
<span class="score-badge">similarity: {src['score']:.3f}</span>
<br><br><em>{src['preview']}</em>
</div>
""", unsafe_allow_html=True)
        if msg.get("latency_ms"):
            st.caption(f"Response time: {msg['latency_ms']}ms | Model: {msg.get('model')} | Chunks: {msg.get('chunks_used')}")

if question := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question, "top_k": 3},
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    st.markdown(data["answer"])

                    with st.expander(f"Sources ({len(data['sources'])} chunks retrieved)"):
                        for src in data["sources"]:
                            st.markdown(f"""
<div class="source-card">
<strong>{src['source']}</strong> — Page {src['page_num']}
<span class="score-badge">similarity: {src['score']:.3f}</span>
<br><br><em>{src['preview']}</em>
</div>
""", unsafe_allow_html=True)

                    st.caption(f"Response time: {data['latency_ms']}ms | Model: {data['model']} | Chunks: {data['chunks_used']}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data["sources"],
                        "latency_ms": data["latency_ms"],
                        "model": data["model"],
                        "chunks_used": data["chunks_used"]
                    })
                else:
                    err = response.json().get("detail", "Something went wrong.")
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {err}"})
            except Exception as e:
                st.error(f"Could not reach API: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})