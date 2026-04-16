import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langfuse import observe, Langfuse, get_client
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from LLM.qwen import chat

langfuse = Langfuse(
    public_key= "",
    secret_key= "",
    host= "https://cloud.langfuse.com"
)

# Konfigurasi Path
DB_PATH = "../chroma_db_wisata"
DOC_PATH = "../documents"

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

@observe
def tanya_wisata_langfuse(query):
    
    langfuse = get_client()
    
    # 1. Retrieval
    relevant_docs = vector_db.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # 2. Memberi nama trace agar mudah dicari di dashboard
    langfuse.update_current_trace(
        name="RAG_Pariwisata_NTT_Borobudur",
        input=query
    )

    # 3. Prompting
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
    
    # 4. Chat ke Local Qwen
    response = chat(prompt)
    
    # Keluar dari fungsi, dekorator otomatis mengirim data ke Langfuse
    return response, context

def evaluasi_dengan_judge():
    test_cases = [
        {"q": "Kapan Borobudur di bangun?", "a": "~770–825 M"},
        {"q": "Berapa besar dimensi Borobudur?", "a": "panjang: 123 meter, lebar: 123 meter, tinggi: 35–42 meter, material: batu andesit"},
        {"q": "Dimana lokasi Borobuduer?", "a": "Borobudur, Magelang, Jawa Tengah, Indonesia"},
        {"q": "Ada berapa arca di borobudur?", "a": "504"}
    ]

    for item in test_cases:
        # PENTING: Panggil fungsi @observe()
        jawaban_ai, context = tanya_wisata_langfuse(item['q'])
        
        # Jalankan Judge
        judge_prompt = f"Bandingkan apakah '{jawaban_ai}' mengandung fakta '{item['a']}'. Jawab 1 (Ya) atau 0 (Tidak). Skor:"
        skor_raw = chat(judge_prompt)
        skor_final = 1 if "1" in skor_raw else 0
        
        # Mengirim skor ke trace yang sedang aktif
        # langfuse_context akan otomatis mencari trace dari fungsi tanya_wisata_langfuse
        langfuse.score(
            name="accuracy",
            value=skor_final,
            comment=f"Ekspektasi: {item['a']}"
        )
        
        print(f"Uji: {item['q']} | Skor: {skor_final}")

if __name__ == "__main__":
    evaluasi_dengan_judge()