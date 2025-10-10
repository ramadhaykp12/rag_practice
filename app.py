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
api_key = os.getenv("GOOGLE_API_KEY")
hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not api_key:
    st.error("GOOGLE_API_KEY tidak ditemukan. Tambahkan ke file .env atau environment variable.")
    st.stop()

# ==== Load PDF ====
# folder penyimpanan dokumen

@st.cache_resource
def download_and_extract():
    url = "https://drive.google.com/drive/folders/1Ur5bXiD3RcNJ-ssodSAZeoDWmmHMv62n?usp=sharing"
    output = "docs.zip"
    if not os.path.exists('docs'):
        gdown.download(url, output, quiet=False)
        with ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall('docs')
        os.remove(output)
        # setelah ekstrak Gabung menjadi satu file
        merged_path = 'docs/merged.pdf'
        with open(merged_path, 'wb') as merged_file:
            for pdf_file in glob.glob('docs/*.pdf'):
                if pdf_file != merged_path:
                    with open(pdf_file, 'rb') as f:
                        merged_file.write(f.read())
        print(f"Merged PDFs into {merged_path}")
    return 'docs'

folder_path = download_and_extract()

# list untuk menyimpan dokumen yang dibaca oleh PyPDFLoader
all_docs = []

for pdf_path in glob.glob(f'{folder_path}/*.pdf'):
    # baca file pdf dengan PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    # muat dokumen
    docs = loader.load()
    # tambahkan dokumen ke list all
    all_docs.extend(docs)

print(f"Loaded {len(all_docs)} documents from {folder_path} folder.")


# ==== Split text ====
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(all_docs)

# ==== Setup Embeddings & VectorStore ====
model = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEndpointEmbeddings(
    model=model,
    task="feature-extraction",
    huggingfacehub_api_token=hf_key,
)

vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("vectorstore")

# ==== Setup LLM ====
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=api_key,
)

# ==== Prompt ====
try:
    prompt = hub.pull("rlm/rag-prompt")
except Exception:
    # fallback manual
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Gunakan konteks berikut untuk menjawab pertanyaan.\n\n{context}"),
        ("human", "{question}")
    ])

# ==== State Definition ====
class State(TypedDict):
    question: str
    context: list[Document]
    answer: str

# ==== Application Steps ====
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=3)
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)
    return {"answer": answer}

# ==== Graph ====
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# ==== Streamlit App ====
st.title("📘 RAG with LangChain & Gemini")
st.write("Ajukan pertanyaan berdasarkan isi PDF `merged.pdf`.")

# Upload CSV file
file = st.file_uploader("Upload file CSV", type=["csv"])
df = pd.read_csv(file) if file else None

if st.button("Submit"):
    if df is not None:
        if "Misi" not in df.columns:
            st.error("Kolom 'Misi' tidak ditemukan di CSV.")
        else:
            for index, row in df.iterrows():
                question = f"Apakah isi {row['Misi']} sesuai dengan pembahasan yang diberikan dari sumber dokumen? Jika ya, berikan jawaban yang ringkas. Jika tidak, berikan penjelasan mengapa tidak sesuai."
                response = graph.invoke({"question": question})
                st.write(f"**Pertanyaan:** {row['Misi']}")
                st.write(f"**Jawaban:** {response['answer']}")
                if index == len(df) - 1:
                    if st.button("Reset"):
                        st.experimental_rerun()
                    
    else:
        st.error("Silakan upload file CSV terlebih dahulu.")
