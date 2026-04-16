import os, sys
# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langfuse import Langfuse, observe, Evaluation
from LLM.qwen import chat
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from LLM.qwen import chat

# 1. INIT LANGFUSE
langfuse = Langfuse(
    public_key="",
    secret_key="",
    host="https://cloud.langfuse.com"
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

    # retrieval + score
    results = vector_db.similarity_search_with_score(query, k=4)

    if not results:
        context = ""
    else:
        context = "\n\n".join([doc.page_content for doc, _ in results])

    # update trace
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

# 2 task func -- kaya students or tanya2
# automatic task di-trace oleh Langfuse
def task_wisata(*, item, **kwargs):
    # Mengambil pertanyaan dari format JSON {"question": "..."}
    query = item.input.get("question") if isinstance(item.input, dict) else item.input
    
    # Panggil logika RAG Anda (pastikan vector_db sudah siap)
    # Kita hanya butuh return jawaban_ai untuk dievaluasi
    jawaban_ai, context = tanya_wisata_langfuse(query)
    
    return jawaban_ai

# 3 evaluator func -- judge
def accuracy_evaluator(*, input, output, expected_output, **kwargs):
    # input: dictionary {"question": "..."}
    # output: jawaban_ai dari task_wisata
    # expected_output: dictionary {"answer": "..."}
    
    kunci = expected_output.get("answer") if isinstance(expected_output, dict) else expected_output
    
    judge_prompt = f"""
    Bandingkan JAWABAN AI dengan REFERENSI. 
    REFERENSI: {kunci}
    JAWABAN AI: {output}
    
    Apakah JAWABAN AI mengandung fakta yang benar sesuai REFERENSI?
    Jawab hanya angka: 1 (Benar) atau 0 (Salah).
    SKOR:"""
    
    skor_raw = chat(judge_prompt).strip()
    skor_final = 1.0 if "1" in skor_raw else 0.0
    
    # return {
    #     # "name": "accuracy",
    #     "value": skor_final,
    #     "comment": f"Expected: {kunci}"
    # }
    return Evaluation(
        name= "accuracy",
        value= skor_final,
        comment= f"Expected: {kunci}"
    )
    
def average_accuracy_evaluator(*, item_results, **kwargs):
    accuracies = [
        eval.value for result in item_results
        for eval in result.evaluations if eval.name == "accuracy"
    ]
    avg = sum(accuracies) / len(accuracies) if accuracies else 0
    return Evaluation(name="avg_accuracy", value=avg)

# 4 RUN EXPERIMENT
def jalankan_uji_coba_dataset(dataset_name):
    print(f"--- Memulai Experiment pada Dataset: {dataset_name} ---")
    
    # get dataset
    dataset = langfuse.get_dataset(dataset_name)
    
    # run experiment
    result = dataset.run_experiment(
        name="Qwen-RAG-Evaluation",
        run_name="Run-BGE-M2-Test1", # nama untuk current session
        description="Evaluasi akurasi RAG pariwisata menggunakan Qwen",
        task=task_wisata,
        evaluators=[accuracy_evaluator],
        run_evaluators= [average_accuracy_evaluator],
        max_concurrency=2 # Batasi agar tidak kena rate limit LLM
    )
    
    print("\n--- Experiment Selesai ---")
    print(f"Dataset Run URL: {result.dataset_run_url}")

if __name__ == "__main__":
    # Pastikan fungsi tanya_wisata_langfuse dan vector_db sudah terdefinisi di atas
    jalankan_uji_coba_dataset("RAG/test-1")