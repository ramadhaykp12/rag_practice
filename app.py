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
    embeddings = HuggingFaceEndpointEmbeddings(model_name="jinaai/jina-embeddings-v4")
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
    st.sidebar.warning("Belum ada dokumen RAG di folder `dokumen/`.")

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
uploaded_file = st.file_uploader("📤 Upload file PDF baru untuk analisis", type=["pdf"])

if uploaded_file:
    pdf_path = os.path.join(uploaded_path, "uploaded.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ File berhasil diunggah!")

    loader = PyPDFLoader(pdf_path)
    new_pdf_docs = loader.load()
    pdf_text = "\n\n".join(doc.page_content for doc in new_pdf_docs)

    if st.button("🚀 Analisis dengan RAG"):
        prompt = (
            "Berdasarkan pengetahuan anda, apakah seluruh isi dokumen berikut "
            "telah tercakup dalam dokumen RAG? Jika ada bagian Perda yang belum tercantum, tuliskan daftarnya.\n\n"
            f"{pdf_text[:4000]}"
        )

        with st.spinner("🔎 Memproses dokumen..."):
            result = graph.invoke({"question": prompt})

        st.subheader("💬 Hasil Analisis:")
        st.write(result["answer"])
