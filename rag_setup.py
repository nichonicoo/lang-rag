import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from parser import parse_txt_to_documents

document_folder = "documents"


# =========================
# LOAD DOCUMENTS
# =========================
def load_documents():
    docs = []

    for filename in os.listdir(document_folder):
        file_path = os.path.join(document_folder, filename)

        # ===== TXT (STRUCTURED) =====
        if filename.endswith(".txt"):
            parsed_docs = parse_txt_to_documents(file_path)
            docs.extend(parsed_docs)

        # ===== PDF =====
        elif filename.endswith(".pdf"):
            from langchain_community.document_loaders import PyMuPDFLoader
            loader = PyMuPDFLoader(file_path)
            pages = loader.load()

            for p in pages:
                if len(p.page_content.strip()) > 30:
                    docs.append(p)

    print(f"[+] Loaded {len(docs)} documents")
    return docs


# =========================
# SETUP RAG
# =========================
def setup_rag():
    docs = load_documents()

    # 🔥 BETTER CHUNKING
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
    )

    chunks = []

    for d in docs:
        pieces = splitter.split_text(d.page_content)

        for i, p in enumerate(pieces):
            if len(p.strip()) < 30:
                continue

            chunks.append(
                Document(
                    page_content=p,
                    metadata={
                        **d.metadata,
                        "chunk_index": i
                    }
                )
            )

    print(f"[+] Total chunks: {len(chunks)}")

    # 🔥 MULTILINGUAL EMBEDDING (IMPORTANT)
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="travel_rag",
    )

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 8,
        },
    )

    print("[✅] RAG Ready")

    return retriever