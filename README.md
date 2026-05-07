# Dartmouth RC RAG (Local - Ollama + Chroma + Hinode mod-llm aware)

A fully local Retrieval-Augmented Generation chat app over:

- Your own files (PDF / TXT / MD / DOCX), and
- Any **Hinode** site, with first-class support for the
  [mod-llm](https://github.com/gethinode/mod-llm) convention. Defaults to
  [rc.dartmouth.edu](https://rc.dartmouth.edu).

Built with **Streamlit + Ollama + ChromaDB + sentence-transformers** with an
optional cross-encoder reranker. No data leaves your machine.

## Why mod-llm?

Hinode sites that enable
[mod-llm](https://gethinode.com/tutorials/generating-llm-content/) publish a
`/llms.txt` index and a clean `/<page>/index.md` for every page - perfect
input for RAG. This app uses those endpoints when available, and gracefully
falls back to `/sitemap.xml` and finally to a same-host crawl (with
`html2text` cleanup).

## Quick start (macOS)

    brew install python git gh ollama
    brew services start ollama
    ollama pull llama3.1:8b

    git clone https://github.com/<your-username>/dartmouth-rc-rag.git
    cd dartmouth-rc-rag
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

    streamlit run app.py

In the browser:

1. Drop in some files **or** click **"Ingest site"** (default: rc.dartmouth.edu).
2. Ask questions. Answers cite sources as `[filename p#]` or `[host/path/ p1]`.

## Files

- `app.py` - Streamlit UI + RAG pipeline
- `hinode_ingest.py` - Hinode mod-llm aware site ingester
- `requirements.txt` - Python dependencies

## License

MIT
