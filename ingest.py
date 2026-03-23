"""Build vector store from documents in data/ (and data/sample_corpus/)."""
import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import DATA_DIR, VECTOR_STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL


def load_documents(data_dir):
    """Load .txt and .pdf files from data_dir and subdirs."""
    documents = []
    data_path = Path(data_dir)
    if not data_path.exists():
        return documents
    for path in data_path.rglob("*"):
        if path.suffix.lower() == ".txt":
            try:
                docs = TextLoader(str(path), encoding="utf-8").load()
                documents.extend(docs)
            except Exception as e:
                print(f"Skip {path}: {e}")
        elif path.suffix.lower() == ".pdf":
            try:
                docs = PyPDFLoader(str(path)).load()
                documents.extend(docs)
            except Exception as e:
                print(f"Skip {path}: {e}")
    return documents


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    sample_dir = os.path.join(DATA_DIR, "sample_corpus")
    os.makedirs(sample_dir, exist_ok=True)
    # Create sample corpus if empty
    sample_file = os.path.join(sample_dir, "welcome.txt")
    if not os.path.isfile(sample_file):
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(
                "Welcome to the knowledge base.\n\n"
                "This chatbot uses RAG: it retrieves relevant passages from this corpus and uses them to answer your questions.\n\n"
                "Context memory is enabled: you can ask follow-up questions and the chat history is kept for the session."
            )
    docs = load_documents(DATA_DIR)
    if not docs:
        print("No documents found in", DATA_DIR)
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(splits, embeddings, persist_directory=VECTOR_STORE_PATH)
    vector_store.persist()
    print(f"Ingested {len(splits)} chunks into {VECTOR_STORE_PATH}")


if __name__ == "__main__":
    main()
