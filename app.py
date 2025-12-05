import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os
from dotenv import load_dotenv
import asyncio, nest_asyncio
import pandas as pd 
import glob
import gdown
from zipfile import ZipFile

# ==== Setup Async ====
nest_asyncio.apply()
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# ==== Load API Key ====
load_dotenv()
api_key = st.secrets['GOOGLE_API_KEY']
hf_key = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

if not api_key:
    st.error("❌ GOOGLE_API_KEY tidak ditemukan. Tambahkan ke file .env atau environment variable.")
    st.stop()

# ==== Folder dokumen ====
folder_path = "dokumen"
uploaded_path = "uploaded_file"
os.makedirs(folder_path, exist_ok=True)

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
    chunks = splitter.split_documents(_all_docs)
    return chunks


@st.cache_resource(show_spinner=False)
def get_embeddings(_hf_key=hf_key):
    model = "Qwen/Qwen3-Embedding-8B"
    embeddings = HuggingFaceEndpointEmbeddings(
        model=model,
        task="feature-extraction",
        huggingfacehub_api_token=_hf_key
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
all_docs = load_documents(folder_path)
if not all_docs:
    st.sidebar.warning("Belum ada dokumen RAG di folder `dokumen/`. Tambahkan dulu PDF sumber RAG.")
chunks = split_documents(all_docs)
embeddings = get_embeddings(hf_key)
vector_store = get_vectorstore(chunks, embeddings)
llm = get_llm(api_key)

# ===============================
# State Definition
# ===============================
class State(TypedDict):
    question: str
    context: list[Document]
    answer: str


# ===============================
# Define Workflow
# ===============================
def retrieve(state: State):
    """Ambil dokumen relevan dari RAG berdasarkan pertanyaan."""
    retrieved_docs = vector_store.similarity_search(state["question"], k=3)
    return {"context": retrieved_docs}


def generate(state: State):
    """Buat jawaban dari konteks dokumen."""
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    full_prompt = f"""
    Berdasarkan sumber pengetahuan RAG yang tersedia,
    berikan analisis terhadap dokumen berikut dan jawab pertanyaan berikut.

    === PDF Content ===
    {docs_content[:4000]}

    === Pertanyaan ===
    {state['question']}

    Berikan jawaban yang jelas dan ringkas.
    """
    response = llm.invoke(full_prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return {"answer": answer}


# ===============================
# Build LangGraph
# ===============================
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph = graph_builder.compile()

# ===============================
# Streamlit UI
# ===============================
st.title("📘 RAG with LangChain & Gemini")
st.write("Analisis dokumen PDF berdasarkan sumber data RAG yang sudah ada.")

# Upload PDF file baru untuk dianalisis
file = st.file_uploader("📤 Upload file PDF baru untuk analisis", type=["pdf"])
if file:
    pdf_path = os.path.join(uploaded_path, "uploaded.pdf")
    with open(pdf_path, "wb") as f:
        f.write(file.getbuffer())
    st.success("✅ File berhasil diunggah!")

    loader = PyPDFLoader(pdf_path)
    new_pdf_docs = loader.load()
    pdf_text = "\n\n".join(doc.page_content for doc in new_pdf_docs)

    if st.button("🚀 Analisis dengan RAG"):
        prompt = (
            f"Berdasarkan pengetahuan anda, apakah seluruh isi dokumen berikut "
            f"tuliskan bagian dari perda yang tidak tercantum dalam dokumen RAG!\n\n{pdf_text[:4000]}"
        )

        with st.spinner("🔎 Memproses dokumen..."):
            result = graph.invoke({"question": prompt})

        st.subheader("💬 Hasil Analisis:")
        st.write(result["answer"])

