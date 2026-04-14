import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from langchain_huggingface import HuggingFaceEmbeddings

# from LLM.gemini_llm import chat
# # , ChatOpenAI

from langchain_google_genai import ChatGoogleGenerativeAI
import os

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key="AIzaSyC0PincVcQMe7XTi2cYKq2IthkzP9U8E54",
)

emdeddings = HuggingFaceEmbeddings(
    model_name="google/embeddinggemma-300m"
)
# llm = ChatOpenAI(model="gpt-4o", temperature=0)

PERSIST_PATH = "./qdrant_db"
COLLECTION_NAME = "star-wars-scripts"


def load_star_wars_script(url, movie_title):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    script_raw = soup.find("pre").get_text()

    return Document(page_content=script_raw, metadata={"title": movie_title})

# def qwen_llm(prompt):
#     return chat(prompt)
# llm = RunnableLambda(qwen_llm)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def main():
    # client = QdrantClient(path=PERSIST_PATH)
    client = QdrantClient(
    url="https://c975af8f-16d9-4b97-a2fd-cca156649512.eu-west-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZGQ3ZjcyNjctNzk3ZC00NzI0LTliMGYtMzc0OThiZTA0ZDVmIn0.2iMsl_Yv647lqFQDzLXaeMOwNXuzDVgGMENFSknr6bI",
)

    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        vectorstore = QdrantVectorStore(
            collection_name=COLLECTION_NAME,
            embeddings=emdeddings,
            client=client,
        )
    except Exception:
        client.close()

        star_wars_scripts = [
            {
                "title": "Star Wars: A New Hope",
                "url": "https://www.imsdb.com/scripts/Star-Wars-A-New-Hope.html",
            },
            {
                "title": "Star Wars: The Empire Strikes Back",
                "url": "https://www.imsdb.com/scripts/Star-Wars-The-Empire-Strikes-Back.html",
            },
            {
                "title": "Star Wars: Return of the Jedi",
                "url": "https://www.imsdb.com/scripts/Star-Wars-Return-of-the-Jedi.html",
            },
        ]

        script_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=250,
            add_start_index=True,
            separators=["\nINT.", "\nEXT.", "\n\n", "\n", " ", ""],
        )

        all_chunks = []

        for script in star_wars_scripts:
            doc = load_star_wars_script(script["url"], script["title"])
            chunks = script_splitter.split_documents([doc])
            all_chunks.extend(chunks)
            print(
                f"Loaded and split script for {script['title']} into {len(chunks)} chunks."
            )

        vectorstore = QdrantVectorStore.from_documents(
            all_chunks,
            embedding=emdeddings,
            path=PERSIST_PATH,
            collection_name=COLLECTION_NAME,
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    template = """
    You are a Star Wars Movie Script Expert. Use ONLY the following script excerpts to answer.
    If the answer is partly contained, provide the best possible answer based on text in the context. 
    If the answer isn't in the context, say "There is no information about this in the original Star Wars scripts.
    Provide a short explanation of how you got the answer based ONLY on the context."

    Context:
    {context}

    Question: 
    {question}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # rag_chain = (
    #     {"context": retriever, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )
    
    rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

    print("\n--- The Star Wars Movie Expert is ready to answer your questions ---")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        docs = retriever.invoke(query)
        
        print("\n--- Retrieved Chunks ---")
        for i, d in enumerate(docs):
            print(f"\n[{i+1}] TITLE:", d.metadata.get("title"))
            print(d.page_content[:300])  # preview 300 char


        response = rag_chain.invoke(query)
        print(f"\nStar Wars Movie Expert: {response}")


if __name__ == "__main__":
    main()