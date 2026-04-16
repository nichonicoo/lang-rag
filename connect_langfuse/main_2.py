import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langfuse import observe, Langfuse
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from LLM.qwen import chat

# ✅ INIT LANGFUSE SEKALI SAJA
langfuse = Langfuse(
    public_key= "pk-lf-35c24d76-1129-4cbe-a528-ba4a53e7d7b4",
    secret_key= "sk-lf-e3681d2e-3661-4aab-8092-ad2f688573c8",
    host= "https://cloud.langfuse.com"
)

DB_PATH = "../chroma_db_wisata"
DOC_PATH = "../documents"

# Embedding
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")


def inisialisasi_rag():
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        print("--- Load Vector DB ---")
        return Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )

    print("--- Create Vector DB ---")

    loader = DirectoryLoader(DOC_PATH, glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print(f"Chunks: {len(chunks)}")
    return vector_db


vector_db = inisialisasi_rag()


@observe()
def tanya_wisata_langfuse(query):

    # ✅ Retrieval + score
    results = vector_db.similarity_search_with_score(query, k=4)

    if not results:
        context = ""
    else:
        context = "\n\n".join([doc.page_content for doc, _ in results])

    # ✅ Update trace
    langfuse.update_current_span(
        name="RAG_Pariwisata",
        input=query
    )

    # ✅ Guard hallucination
    if not context.strip():
        return "Maaf, saya tidak menemukan informasi di database.", context

    prompt = f"""
    Anda adalah asisten pariwisata Indonesia.

    Gunakan HANYA data berikut:
    {context}

    Jika jawaban tidak ada → jawab: "Saya tidak tahu"

    Pertanyaan:
    {query}

    Jawaban:
    """

    response = chat(prompt)

    return response, context


def evaluasi_dengan_judge():
    test_cases = [
        {"q": "Kapan Borobudur dibangun?", "a": "770"},
        {"q": "Dimana lokasi Borobudur?", "a": "Magelang"},
        {"q": "Ada berapa arca di Borobudur?", "a": "504"}
    ]

    for item in test_cases:
        jawaban_ai, context = tanya_wisata_langfuse(item["q"])

        judge_prompt = f"""
        Apakah jawaban berikut mengandung fakta '{item['a']}'?

        Jawaban AI:
        {jawaban_ai}

        Jawab hanya:
        1 = YA
        0 = TIDAK
        """

        skor_raw = chat(judge_prompt).strip()

        # ✅ lebih strict
        skor_final = 1 if skor_raw.startswith("1") else 0

        langfuse.score_current_span(
            name="accuracy",
            value=skor_final,
            comment=f"Expected: {item['a']}"
        )

        print(f"Q: {item['q']}")
        print(f"A: {jawaban_ai}")
        print(f"Score: {skor_final}")
        print("-" * 40)

# def run_evaluation_from_langfuse_dataset(dataset_name):
#     # 1. Tarik dataset dari Langfuse
#     # dataset = langfuse.get_dataset(dataset_name)
#     dataset = Langfuse.get_dataset(dataset_name)
    
#     print(f"\n--- Menjalankan Evaluasi dari Dataset: {dataset_name} ---")

#     for item in dataset.items:
#         # 2. Jalankan RAG menggunakan 'input' dari dataset
#         # 'item.input' adalah pertanyaan yang tersimpan di Langfuse
#         jawaban_ai, langfuse     = tanya_wisata_langfuse(item.input)

#         # Ambil trace_id untuk menghubungkan hasil ke dataset
#         trace_id = langfuse.get_current_trace_id()
        
#         # 3. Judge menggunakan Qwen
#         judge_prompt = f"""
#         Bandingkan JAWABAN AI dengan REFERENSI. 
#         Apakah JAWABAN AI mengandung fakta yang benar dari REFERENSI?
#         REFERENSI: {item.expected_output}
#         JAWABAN AI: {jawaban_ai}
        
#         Jawab hanya dengan angka: 1 (Benar) atau 0 (Salah)."""
        
#         skor_raw = chat(judge_prompt)
#         skor_final = 1 if "1" in str(skor_raw) else 0

#         # 4. Link-kan hasil ini ke Dataset di Langfuse (PENTING!)
#         item.link(
#             trace_id=trace_id, 
#             run_name="Qwen-BGE-M3-K4-Run" # Beri nama eksperimen Anda
#         )

#         # 5. Kirim Skor
#         langfuse.score(
#             trace_id=trace_id,
#             name="accuracy",
#             value=skor_final,
#             comment=f"Expected: {item.expected_output}"
#         )
        
#         print(f"Uji: {item.input} | Skor: {skor_final}")


def run_evaluation_from_langfuse_dataset(dataset_name):
    # Ambil dataset dari cloud
    dataset = langfuse.get_dataset(dataset_name)
    
    print(f"\n--- Menjalankan Evaluasi: {dataset_name} ---")

    for item in dataset.items:
        # Handling format JSON {"question": "..."}
        pertanyaan = item.input.get("question") if isinstance(item.input, dict) else item.input
        kunci_jawaban = item.expected_output.get("answer") if isinstance(item.expected_output, dict) else item.expected_output

        # Jalankan RAG
        jawaban_ai, context_docs = tanya_wisata_langfuse(pertanyaan)

        # ✅ AMBIL TRACE ID (Cara versi terbaru)
        # Jika langfuse_context masih ada tapi berubah method:
        trace_id = langfuse.get_current_trace_id() 
        
        # 1. Hubungkan ke Dataset Run
        item.run(
            trace_id=trace_id, 
            run_name="Run-Qwen-BGE-M3" 
        )
        
        # dataset.run_experiment(name= "Run-Qwen-BGE-M2", run_name="Run-Qwen-BGE-M2", task=)

        # 2. Judge (LLM-as-a-judge)
        # judge_prompt = f"Kunci: {kunci_jawaban}\nAI: {jawaban_ai}\nApakah benar? Jawab 1/0:"
        judge_prompt = f"""
        Bandingkan JAWABAN AI dengan REFERENSI. 
        Apakah JAWABAN AI mengandung fakta yang benar dari REFERENSI?
        REFERENSI: {item.expected_output}
        JAWABAN AI: {jawaban_ai}
        
        Jawab hanya dengan angka: 1 (Benar) atau 0 (Salah)."""
        skor_raw = chat(judge_prompt).strip()
        skor_final = 1 if "1" in skor_raw else 0

        # 3. Kirim Skor menggunakan client utama
        langfuse.score_current_trace(
            trace_id=trace_id,
            name="accuracy",
            value=skor_final,
            comment=f"Expected: {kunci_jawaban}"
        )
        
        print(f"Q: {pertanyaan} | Score: {skor_final}")

# if __name__ == "__main__":
#     # Pastikan nama dataset sesuai dengan yang Anda buat di dashboard
#     run_evaluation_from_langfuse_dataset("wisata-indonesia-bench")

if __name__ == "__main__":
    # evaluasi_dengan_judge() 
    # with langfuse datasets 
    run_evaluation_from_langfuse_dataset("RAG/test-1")
    # dataset = langfuse.get_dataset("RAG/test-1")
    # print(dataset)