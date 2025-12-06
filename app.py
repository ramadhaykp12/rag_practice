import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os
from dotenv import load_dotenv
import asyncio, nest_asyncio
import glob

# ==== Setup Async ====
nest_asyncio.apply()
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ==== Load API Key ====
load_dotenv()
api_key = st.secrets.get("GOOGLE_API_KEY")
hf_key = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

if not api_key:
    st.error("❌ GOOGLE_API_KEY tidak ditemukan di secrets.")
    st.stop()

if not hf_key:
    st.error("❌ HUGGINGFACEHUB_API_TOKEN tidak ditemukan di secrets.")
    st.stop()

# ==== Folder dokumen ====
folder_path = "dokumen"
uploaded_path = "uploaded_file"
os.makedirs(folder_path, exist_ok=True)
os.makedirs(uploaded_path, exist_ok=True)

# ===============================
# Fungsi Helper
# ===============================
@st.cache_data(show_spinner=False)
def load_documents(folder_path="dokumen"):
    """Membaca semua PDF di folder dokumen."""
    all_docs = []
    for pdf_path in glob.glob(f"{folder_path}/*.pdf"):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs


@st.cache_resource(show_spinner=False)
def split_documents(_all_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(_all_docs)


@st.cache_resource(show_spinner=False)
def get_embeddings(_hf_key=hf_key):
    # 🔥 Model embedding GRATIS & MULTILINGUAL
    model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    embeddings = HuggingFaceEndpointEmbeddings(
        model=model,
        task="feature-extraction",
        huggingfacehub_api_token=_hf_key,
    )
    return embeddings


@st.cache_resource(show_spinner=False)
def get_vectorstore(_chunks, _embeddings):
    os.makedirs("vectorstore", exist_ok=True)
    index_path = "vectorstore/faiss_index"
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, _embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(_chunks, _embeddings)
        vectorstore.save_local(index_path)
        return vectorstore


@st.cache_resource(show_spinner=False)
def get_llm(_api_key=api_key):
    """Inisialisasi LLM Gemini"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=_api_key,
    )


# ===============================
# Bangun Knowledge Base RAG
# ===============================
st.sidebar.header("📚 Building Knowledge Base")

all_docs = load_document
