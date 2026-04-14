import re
from langchain_core.documents import Document

def parse_txt_to_documents(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # ===== METADATA HEADER =====
    metadata_block = re.search(r"---(.*?)---", text, re.DOTALL)
    metadata = {}

    if metadata_block:
        lines = metadata_block.group(1).strip().split("\n")
        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                metadata[k.strip()] = v.strip()

    # ===== SPLIT SECTION =====
    sections = re.split(r"=== (.*?) ===", text)

    documents = []

    for i in range(1, len(sections), 2):
        section_title = sections[i].strip()
        content = sections[i + 1].strip()

        if not content:
            continue

        doc = Document(
            page_content=content,
            metadata={
                **metadata,
                "section": section_title,
                "source": file_path
            }
        )

        documents.append(doc)

    return documents