import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from LLM.qwen import chat # Mengambil fungsi chat dari file qwen.py Anda

# 1. SETUP PEMBACAAN DOKUMEN
print("Membaca dokumen dari folder documents...")
# Menggunakan DirectoryLoader untuk membaca semua file .txt di folder documents
loader = DirectoryLoader("./documents", glob="*.txt", loader_cls=TextLoader)
docs = loader.load()

# 2. CHUNKING
# Memecah dokumen menjadi potongan 1000 karakter dengan overlap 150
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(docs)

# 3. EMBEDDING & VECTOR DB
# Menggunakan model lokal agar tidak perlu internet
print("Membuat Vector Database...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
vector_db = Chroma.from_documents(chunks, embeddings)

def tanya_wisata(query):
    # 4. RETRIEVAL: Cari potongan teks paling relevan
    relevant_docs = vector_db.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    print('context: ', context)
    
    # 5. PROMPT ENGINEERING
    # Kita rakit pesan untuk Qwen agar menjawab berdasarkan konteks
    prompt = f"""Anda adalah asisten pariwisata Indonesia. 
Gunakan data di bawah ini untuk menjawab pertanyaan. 
Jika tidak ada di data, katakan Anda tidak tahu.

DATA KONTEKS:
{context}

PERTANYAAN: 
{query}

JAWABAN:"""

    # 6. KIRIM KE QWEN (Fungsi di qwen.py)
    response = chat(prompt)
    return response

# Contoh Penggunaan
if __name__ == "__main__":
    pertanyaan = "Siapa Pemilik Danau Toba?"
    jawaban = tanya_wisata(pertanyaan)
    print(f"\nQwen menjawab:\n{jawaban}")