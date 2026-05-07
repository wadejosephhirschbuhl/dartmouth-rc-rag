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

## Performance / Dont melt your machine

The sidebar has a **Performance** section that exposes Ollamas main
throttling knobs. All three are passed straight through to `ollama.chat()`
so changes take effect on the next question.

### CPU threads (`num_thread`)

Caps how many CPU threads Ollama uses. Defaults to half your cores. Lower =
cooler and quieter, slightly slower.

### GPU layers (`num_gpu`)

LLMs are made of stacked transformer layers (llama3.1:8b has 32, llama3.2:3b
has 28). `num_gpu` controls how many of those layers run on the GPU, counted
from the top:

| Value | What happens                              | Speed       | Heat       |
|-------|-------------------------------------------|-------------|------------|
| `-1`  | All layers on GPU (max offload)           | Fastest     | Hottest    |
| `0`   | All layers on CPU (no GPU used at all)    | Slowest     | Coolest    |
| `8`   | First 8 layers on GPU, rest on CPU        | Medium      | Medium     |
| `16`  | First 16 layers on GPU                    | Faster      | Hotter     |
| `24+` | Most/all on GPU                           | Near `-1`   | Near `-1`  |

`-1` is "use them all" - works for any model regardless of layer count.
On Apple Silicon, GPU and CPU share unified memory, so `-1` is almost
always the right choice unless youre actively trying to throttle.
On NVIDIA, use a partial value (e.g. `16`) only if the full model wont
fit in VRAM.

### Context window (`num_ctx`)

Smaller = far less RAM and compute. `2048` is plenty for most RAG queries;
default is `4096`. Halving the context roughly halves memory pressure.

### Gentle mode

Click the **"Gentle mode (safe defaults)"** button to apply a cool-and-quiet
preset:

- `num_thread` = half your cores
- `num_gpu` = `0` (CPU only)
- `num_ctx` = `2048`
- Switches the model to `llama3.2:3b`

Pull that model first if you havent:

    ollama pull llama3.2:3b

### Embedding device

The sentence-transformers embedding model runs on CPU by default. Override
with the `RAG_EMBED_DEVICE` env var (`mps` for Apple GPU, `cuda` for NVIDIA):

    RAG_EMBED_DEVICE=mps streamlit run app.py

### Other heat-reducing options

- **Smaller model**: `llama3.2:3b` or `llama3.2:1b` instead of `llama3.1:8b`.
- **Leave the cross-encoder reranker off** unless you need it.
- **macOS Low Power Mode** (System Settings -> Battery) caps GPU clock globally.
- Run Ollama under `cpulimit` or `taskpolicy -c utility` for hard system caps:

        brew install cpulimit
        brew services stop ollama
        cpulimit -l 200 -- ollama serve         # max ~2 cores total
        # or
        taskpolicy -c utility ollama serve      # prefer efficiency cores

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
