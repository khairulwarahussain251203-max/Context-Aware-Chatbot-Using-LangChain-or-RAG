# Task-2: Context-Aware Chatbot Using LangChain / RAG

Conversational chatbot with context memory and retrieval from a vectorized document store, deployed with Streamlit.

## Features

- **RAG**: Retrieve answers from a vector store (ChromaDB) over your corpus.
- **Context memory**: Conversational history kept in session for follow-up questions.
- **Streamlit UI**: Deploy and chat in the browser.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Add documents to `data/` (e.g. PDFs or `.txt` files) or use the sample corpus in `data/sample_corpus/`.
3. Set `OPENAI_API_KEY` in `.env` (or use a local LLM via LangChain integrations).

## Usage

```bash
# Ingest documents into the vector store (first time or when docs change)
python ingest.py

# Run the chatbot
streamlit run app.py
```

## Configuration

- Edit `config.py` to set embedding model, chunk size, and retriever top-k.
- For local LLM, swap the ChatOpenAI model in `app.py` for your provider (e.g. Ollama, HuggingFace).
