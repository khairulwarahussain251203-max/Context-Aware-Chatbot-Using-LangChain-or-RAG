"""Streamlit app: context-aware RAG chatbot with conversation memory."""
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

from config import VECTOR_STORE_PATH, EMBEDDING_MODEL, TOP_K_RETRIEVAL, USE_OPENAI


@st.cache_resource
def get_vector_store():
    if not Path(VECTOR_STORE_PATH).exists():
        return None
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)


@st.cache_resource
def get_llm():
    if USE_OPENAI:
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    try:
        return Ollama(model="llama2")
    except Exception:
        return None


def _format_chat_history(msgs, last_n=5):
    lines = []
    for m in msgs.messages[-last_n:]:
        role = "User" if m.type == "human" else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines) if lines else ""


def main():
    st.set_page_config(page_title="Context-Aware RAG Chatbot", page_icon="💬")
    st.title("Context-Aware RAG Chatbot")
    st.caption("Answers using your documents + conversation memory")

    vector_store = get_vector_store()
    llm = get_llm()
    if vector_store is None:
        st.info("Run `python ingest.py` first to build the vector store from documents in `data/`.")
        return
    if llm is None:
        st.info("Set OPENAI_API_KEY in .env or run Ollama with a model (e.g. llama2) for the LLM.")
        return

    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
    msgs = StreamlitChatMessageHistory(key="chat_messages")

    PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Use the following context to answer the question. "
            "If the answer is not in the context, say so.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        ),
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
    )

    for msg in msgs.messages:
        with st.chat_message(msg.type):
            st.markdown(msg.content)

    if prompt := st.chat_input("Ask about your documents (context is remembered):"):
        msgs.add_user_message(prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                chat_history = _format_chat_history(msgs)
                query = f"{chat_history}\n\nCurrent question: {prompt}" if chat_history else prompt
                result = qa_chain.invoke({"query": query})
                answer = result.get("result", "No answer generated.")
                msgs.add_ai_message(answer)
                st.markdown(answer)
            except Exception as e:
                st.error(str(e))
                msgs.add_ai_message(str(e))
