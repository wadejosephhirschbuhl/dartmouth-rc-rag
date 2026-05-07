# Dartmouth RC RAG (Local - Ollama + Chroma + Hinode mod-llm aware)

A fully local Retrieval-Augmented Generation chat app over:

- Your own files (PDF / TXT / MD / DOCX), and
- Any **Hinode** site, with first-class support for the
  [mod-llm](https://github.com/gethinode/mod-llm) convention. Defaults to
  [rc.dartmouth.edu](https://rc.dartmouth.edu).

Built with **Streamlit + Ollama + ChromaDB + sentence-transformers** with an
optional cross-encoder reranker. No data leaves your machine.

## How site ingestion works

The app discovers pages in this order, automatically falling back as needed:

1. `/llms.txt` (Hinode mod-llm) - clean, structured index
2. `/sitemap.xml` - any Hugo / static site
3. Same-host BFS crawl - last resort

For each discovered page it then tries `/<page>/index.md` (mod-llm) before
falling back to HTML -> markdown via `html2text` + BeautifulSoup. So the same
codebase works on a fully mod-llm-enabled Hinode site (best quality) and on a
Hinode site that has not enabled mod-llm yet (works fine via sitemap).

Tested live against:

- `https://gethinode.com` -> `mode: llms.txt`, 110 pages, 0 skipped
- `https://rc.dartmouth.edu` -> `mode: sitemap.xml`, ~200 pages
  (Dartmouth has not enabled mod-llm yet; will auto-upgrade to `llms.txt` mode
  the moment they do.)

## Requirements

- macOS, Linux, or Windows
- **Python 3.12** (3.10 or 3.11 also fine; see Tahoe note below before using 3.14)
- [Ollama](https://ollama.com) running locally
- ~6 GB free disk for the LLM weights and Python deps

## Quick start (macOS)

    # 1. Install prerequisites
    brew install python@3.12 git gh ollama
    brew services start ollama
    ollama pull llama3.1:8b

    # 2. Clone and set up
    git clone https://github.com/wadejosephhirschbuhl/dartmouth-rc-rag.git
    cd dartmouth-rc-rag

    # Use Python 3.12 explicitly (see Tahoe note below)
    /opt/homebrew/bin/python3.12 -m venv .venv
    source .venv/bin/activate

    pip install --upgrade pip
    pip install -r requirements.txt

    # 3. Run
    streamlit run app.py

In the browser at http://localhost:8501:

1. Drop in some files **or** click **"Ingest site"** (default: rc.dartmouth.edu).
2. Ask questions. Answers cite sources as `[filename p#]` or `[host/path/ p1]`.

## macOS Tahoe (macOS 26) note

If you are on macOS Tahoe and use Homebrew Python, you may hit this error
when pip-installing or running the app:

    ImportError: dlopen(.../pyexpat.cpython-3XX-darwin.so):
    Symbol not found: _XML_SetAllocTrackerActivationThreshold
    Expected in: /usr/lib/libexpat.1.dylib

This is a dynamic-linker issue: macOS Tahoe ships an older `libexpat` in
`/usr/lib/`, and the linker prefers it over Homebrews newer `libexpat`,
which Homebrew Python was built against.

Fixes (try in order):

1. Reinstall expat and Python:

        brew install expat
        brew reinstall python@3.12

2. If that does not work, build Python from source so it embeds the absolute
   path to Homebrews expat:

        brew uninstall --ignore-dependencies python@3.12
        brew install --build-from-source python@3.12

3. Or skip Homebrew Python entirely and use the official python.org
   installer for Python 3.12 from <https://www.python.org/downloads/macos/>,
   then create the venv with:

        /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -m venv .venv

Avoid Python 3.14 on Tahoe for now - the same expat issue affects it and
not all of our deps have stable 3.14 wheels yet.

## Configuration

Sidebar controls:

- **Collection name** - which Chroma collection to read/write
  (default `rc_website`)
- **Ollama model** - any model you have pulled (default `llama3.1:8b`)
- **Top-k**, **Temperature**, **Max context chars** - retrieval / generation knobs
- **Cross-encoder reranker** (optional) - slower but higher quality
- **Chunk size / overlap** - applied at ingest time

Environment variables:

- `OLLAMA_CHAT_MODEL` - override the default chat model
- `RAG_RERANK_MODEL` - override the cross-encoder

## Files

- `app.py` - Streamlit UI + RAG pipeline
- `hinode_ingest.py` - Hinode mod-llm aware site ingester
- `requirements.txt` - Python dependencies

## Privacy

Everything runs on your machine. Ollama serves the LLM locally on
`http://127.0.0.1:11434`, embeddings are computed locally with
`sentence-transformers/all-MiniLM-L6-v2`, and the vector store is a local
Chroma DB at `./rag_chroma_db/`. Nothing is sent to a third party.

## License

MIT - see [LICENSE](LICENSE).
