from rag_setup import setup_rag

retriever = setup_rag()

query = "Danau Toba adalah ?"

docs = retriever.invoke(query)

print('question: ', query)

for d in docs:
    
    print("\n---")
    print("SECTION:", d.metadata.get("section"))
    print("TEXT:", d.page_content[:200])