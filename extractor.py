import requests, re
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

URL = "https://id.wikipedia.org/wiki/Nusa_Tenggara_Timur?action=render"
OUTPUT_FILE = "ntt_full.txt"

def extract_infobox(soup):
    infobox = soup.select_one("table.infobox")
    data = {}

    if not infobox:
        print("❌ infobox not found")
        return data

    rows = infobox.select("tr")

    for row in rows:
        header = row.select_one("th.infobox-label")
        value = row.select_one("td.infobox-data")

        # skip kalau bukan data row
        if not header or not value:
            continue

        key = header.get_text(" ", strip=True)
        key = re.sub(r"\s+", " ", key)
        key = key.replace("•", "").strip()

        val = value.get_text(" ", strip=True)
        val = re.sub(r"\s+", " ", val)

        # skip kalau kosong
        if not val:
            continue

        data[key] = val

    return data

def extract_wikipedia_clean(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    response = requests.get(url, headers=headers)
    print("Status:", response.status_code)
    
    print('hasil: ', response.text[:500])

    soup = BeautifulSoup(response.content, "html.parser")

    content_div = soup.select_one("div.mw-parser-output")

    if not content_div:
        print("❌ content_div not found!")
        return ""

    paragraphs = content_div.find_all("p")
    print("Total paragraphs:", len(paragraphs))
    
    

    texts = []
    for p in paragraphs:
        txt = p.get_text().strip()

        if not txt:
            continue

        txt = re.sub(r"\[\d+\]", "", txt)

        texts.append(txt)

    return "\n\n".join(texts)

def extract_all_sections(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    soup = BeautifulSoup(requests.get(url, headers=headers).content, "html.parser")
    
    infobox_data = extract_infobox(soup)

    data = {"INFOBOX": infobox_data}
    current_section = "intro"

    for tag in soup.find_all(["h2","h4", "p"]):
        # if tag.name == "h2":
        #     current_section = tag.get_text().strip()
        #     data[current_section] = []
        # elif tag.name == "p":
        #     txt = tag.get_text().strip()
        #     if txt:
        #         txt = re.sub(r"\[\d+\]", "", txt)
        #         data.setdefault(current_section, []).append(txt)
        if tag.name == "h2":
            section_title = re.sub(r"\[.*?\]", "", tag.get_text().strip())
            current_section = section_title
            data[current_section] = []
            
        elif tag.name == "h4":
            section_title = re.sub(r"\[.*?\]", "", tag.get_text().strip())
            current_section = section_title
            data[current_section] = []

        elif tag.name == "p":
            txt = tag.get_text().strip()

            if not txt:
                continue

            txt = re.sub(r"\[\d+\]", "", txt)
            data.setdefault(current_section, []).append(txt)

    return data



def save_to_txt(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for section, paragraphs in data.items():
            f.write(f"=== {section} ===\n\n")

            for p in paragraphs:
                f.write(p + "\n\n")

            f.write("\n")

    print(f"✅ Saved to {filename}")

# text = extract_all_sections(URL)

# print("OUTPUT LENGTH:", len(text))
# print(text)

def main():
    print("🔄 Scraping Wikipedia...")

    data = extract_all_sections(URL)

    print("📊 Sections found:", list(data.keys()))

    save_to_txt(data, OUTPUT_FILE)


if __name__ == "__main__":
    main()