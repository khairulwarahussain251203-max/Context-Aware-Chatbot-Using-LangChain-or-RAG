# Context-Aware Chatbot (LangChain / RAG)

## 📋 Task Overview

This project implements a **context-aware conversational assistant** using **Retrieval-Augmented Generation (RAG)**. Documents are ingested, chunked, embedded, and stored in a **vector database (ChromaDB)**. At query time, the most relevant passages are retrieved and passed to an **LLM** together with the user’s question and recent **chat history**, so answers stay grounded in your corpus while supporting natural follow-ups.

## 🎯 Objective

As part of the **AI/ML Engineering Internship at DevelopersHub Corporation**, this task demonstrates:

- **Document ingestion & chunking**: Loading `.txt` / PDF files and splitting them with overlap for better context capture.
- **Semantic retrieval**: Embedding text with **sentence-transformers** and querying Chroma for top‑k similar chunks.
- **Grounded generation**: Using **LangChain** `RetrievalQA` so the model conditions on retrieved context before answering.
- **Session memory**: Keeping **Streamlit** chat history so follow-up questions reuse conversational context in the query.
- **Flexible LLM backends**: **OpenAI** (`ChatOpenAI`) when `OPENAI_API_KEY` is set, otherwise **Ollama** (e.g. `llama2`) for local inference.

## 📊 Data Specifications

| Item | Description |
|------|-------------|
| **Source** | Your own files under `data/` (including `data/sample_corpus/`). |
| **Formats** | Plain text (`.txt`) and PDF (`.pdf`). |
| **Processing** | Recursive split with configurable `CHUNK_SIZE` and `CHUNK_OVERLAP` in `config.py`. |
| **Storage** | Embeddings + metadata persisted under `chroma_db/` (local Chroma persist directory). |
| **Retrieval** | Top‑`k` passages (default `TOP_K_RETRIEVAL` in `config.py`) fed into the “stuff” QA chain. |

## 🛠️ Tech Stack

| Layer | Technology |
|--------|------------|
| **Language** | Python 3.x (3.10+ recommended) |
| **UI** | Streamlit |
| **Orchestration** | LangChain (`RetrievalQA`, prompts, chat history) |
| **Vector store** | ChromaDB |
| **Embeddings** | Hugging Face / sentence-transformers (`all-MiniLM-L6-v2` by default) |
| **LLM** | OpenAI API or Ollama |
| **Documents** | `pypdf`, directory loaders |
| **Config** | `python-dotenv` (`.env` for secrets) |

## 🔄 Pipeline Architecture

### 1. Ingestion & indexing (`ingest.py`)

1. **Load** documents from `data/` (recursive).
2. **Split** into overlapping chunks (`RecursiveCharacterTextSplitter`).
3. **Embed** each chunk with the configured embedding model.
4. **Persist** vectors and text in **Chroma** at `chroma_db/`.

### 2. Query & generation (`app.py`)

1. **Retrieve** top‑k chunks for the user query (optionally prefixed with recent chat turns).
2. **Stuff** retrieved context into a single prompt with the current question.
3. **Generate** an answer with the selected LLM (OpenAI or Ollama).
4. **Display** messages in a Streamlit chat UI and append to session history.

## 📈 Design & Behavior Summary

| Component | Role |
|-----------|------|
| **Chunking** | Balances context length vs. granularity; overlap reduces boundary artifacts. |
| **MiniLM embeddings** | Fast, local-friendly embeddings suitable for CPU development. |
| **Stuff chain** | Simple, effective when combined context fits in the model context window. |
| **History in query** | Improves follow-ups (“What about the second point?”) at the cost of longer prompts—use a capable LLM for best results. |

*Quantitative accuracy depends on your documents, questions, and LLM choice; tune `TOP_K_RETRIEVAL`, chunking, and the model for your use case.*

## 🚀 Installation & Usage

### 1. Clone & install

```bash
git clone https://github.com/khairulwarahussain251203-max/Context-Aware-Chatbot-Using-LangChain-or-RAG.git
cd Context-Aware-Chatbot-Using-LangChain-or-RAG
pip install -r requirements.txt
```

### 2. Environment

Create a `.env` file in the project root (do **not** commit it):

```env
OPENAI_API_KEY=your-key-here
```

If you prefer **Ollama**, omit the key (or leave it unset), install [Ollama](https://ollama.com), pull a model (e.g. `llama2`), and ensure it is running.

### 3. Ingest documents

```bash
python ingest.py
```

Run again whenever you add or change files under `data/`.

### 4. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal and chat in the browser.

## 📂 Project Structure

```
├── app.py              # Streamlit UI + retrieval QA + chat memory
├── ingest.py           # Document loading, chunking, Chroma indexing
├── config.py           # Paths, chunk sizes, embedding model, retrieval k
├── requirements.txt    # Python dependencies
├── data/               # Your corpus (e.g. sample_corpus/welcome.txt)
├── chroma_db/          # Generated vector store (typically gitignored)
├── .env                # Local secrets (gitignored)
└── README.md
```

## 📊 Key Insights

- **Grounding**: Answers should cite behavior implied by retrieved chunks; if context is missing, the prompt asks the model to say so.
- **Session context**: Recent user/assistant lines are prepended to the query so follow-up questions stay coherent.
- **Operational note**: Smaller local models may need shorter prompts or fewer history turns; cloud models generally handle longer context better.

## 🔧 Future Improvements

### 1. Model & retrieval

- **Hybrid search** (keyword + dense) for better recall on exact terms.
- **Re-ranking** retrieved chunks with a cross-encoder.
- **Larger / instruct-tuned LLMs** for stricter instruction following.
- **Source citations** (page/chunk id) in the UI.

### 2. Deployment & MLOps

- **API layer**: Expose retrieval + generation via **FastAPI** or **Flask**.
- **Auth & rate limits** for a shared deployment.
- **Containerization** (Docker) for reproducible runs.
- **Observability**: Log queries, latencies, and retrieval scores; optional feedback buttons in Streamlit.

## Author

Developed as part of the **AI/ML Engineering Internship at DevelopersHub Corporation**.

## 📄 License

This project is for **educational purposes** as part of the internship program.
