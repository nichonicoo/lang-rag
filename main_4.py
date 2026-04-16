# add for chromaDB
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from LLM.qwen import chat 

# Konfigurasi Path
DB_PATH = "./chroma_db_wisata"
DOC_PATH = "./documents"

# 1. Inisialisasi Model Embedding
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

def inisialisasi_rag():
    # Cek apakah database sudah ada di folder
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        print("--- Memuat Database Vektor dari Harddisk ---")
        vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    else:
        print("--- Membuat Database Vektor Baru (Proses Embedding) ---")
        loader = DirectoryLoader(DOC_PATH, glob="*.txt", loader_cls=TextLoader)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(docs)
        
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=DB_PATH
        )
        print(f"Berhasil memproses {len(chunks)} chunks.")
    
    return vector_db

# Panggil fungsi inisialisasi
vector_db = inisialisasi_rag()

def tanya_wisata(query):
    # Search tetap sama
    relevant_docs = vector_db.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = f"""
    Anda adalah asisten pariwisata Indonesia. 
    Gunakan data di bawah ini untuk menjawab pertanyaan. 
    Jika tidak ada di data, katakan Anda tidak tahu.
    DATA KONTEKS:
    {context}

    PERTANYAAN: 
    {query}
    
    JAWABAN:
    """
    # Gunakan konteks ini: {context}\n\nPertanyaan: {query}"""
    return chat(prompt)

if __name__ == "__main__":
    print(tanya_wisata("Apa perbedaan status UNESCO antara Danau Toba dan Candi Borobudur?"))