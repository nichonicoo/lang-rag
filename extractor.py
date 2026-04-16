# import requests, re
# from bs4 import BeautifulSoup
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# URL = "https://id.wikipedia.org/wiki/Sulawesi_Utara?action=render"
# OUTPUT_FILE = "sulut_full.txt"

# def extract_infobox(soup):
#     infobox = soup.select_one("table.infobox")
#     data = {}

#     if not infobox:
#         print("❌ infobox not found")
#         return data

#     rows = infobox.select("tr")

#     for row in rows:
#         header = row.select_one("th.infobox-label")
#         value = row.select_one("td.infobox-data")

#         # skip kalau bukan data row
#         if not header or not value:
#             continue

#         key = header.get_text(" ", strip=True)
#         key = re.sub(r"\s+", " ", key)
#         key = key.replace("•", "").strip()

#         val = value.get_text(" ", strip=True)
#         val = re.sub(r"\s+", " ", val)

#         # skip kalau kosong
#         if not val:
#             continue

#         data[key] = val

#     return data

# def extract_wikipedia_clean(url):
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
#     }

#     response = requests.get(url, headers=headers)
#     print("Status:", response.status_code)
    
#     print('hasil: ', response.text[:500])

#     soup = BeautifulSoup(response.content, "html.parser")

#     content_div = soup.select_one("div.mw-parser-output")

#     if not content_div:
#         print("❌ content_div not found!")
#         return ""

#     paragraphs = content_div.find_all("p")
#     print("Total paragraphs:", len(paragraphs))
    
    

#     texts = []
#     for p in paragraphs:
#         txt = p.get_text().strip()

#         if not txt:
#             continue

#         txt = re.sub(r"\[\d+\]", "", txt)

#         texts.append(txt)

#     return "\n\n".join(texts)

# def extract_all_sections(url):
#     headers = {"User-Agent": "Mozilla/5.0"}
#     soup = BeautifulSoup(requests.get(url, headers=headers).content, "html.parser")
    
#     infobox_data = extract_infobox(soup)

#     data = {"INFOBOX": infobox_data}
#     current_section = "intro"

#     for tag in soup.find_all(["h2","h4", "p"]):
#         # if tag.name == "h2":
#         #     current_section = tag.get_text().strip()
#         #     data[current_section] = []
#         # elif tag.name == "p":
#         #     txt = tag.get_text().strip()
#         #     if txt:
#         #         txt = re.sub(r"\[\d+\]", "", txt)
#         #         data.setdefault(current_section, []).append(txt)
#         if tag.name == "h2":
#             section_title = re.sub(r"\[.*?\]", "", tag.get_text().strip())
#             current_section = section_title
#             data[current_section] = []
            
#         elif tag.name == "h4":
#             section_title = re.sub(r"\[.*?\]", "", tag.get_text().strip())
#             current_section = section_title
#             data[current_section] = []

#         elif tag.name == "p":
#             txt = tag.get_text().strip()

#             if not txt:
#                 continue

#             txt = re.sub(r"\[\d+\]", "", txt)
#             data.setdefault(current_section, []).append(txt)

#     return data



# def save_to_txt(data, filename):
#     with open(filename, "w", encoding="utf-8") as f:
#         for section, paragraphs in data.items():
#             f.write(f"=== {section} ===\n\n")

#             for p in paragraphs:
#                 f.write(p + "\n\n")

#             f.write("\n")

#     print(f"✅ Saved to {filename}")

# # text = extract_all_sections(URL)

# # print("OUTPUT LENGTH:", len(text))
# # print(text)

# def main():
#     print("🔄 Scraping Wikipedia...")

#     data = extract_all_sections(URL)

#     print("📊 Sections found:", list(data.keys()))

#     save_to_txt(data, OUTPUT_FILE)


# if __name__ == "__main__":
#     main()


import requests
import re
from bs4 import BeautifulSoup

URL = "https://id.wikipedia.org/wiki/Jawa_Tengah?action=render"
OUTPUT_FILE = "jateng_full.txt"

def extract_infobox(soup):
    """
    Mengekstraksi data dari tabel infobox Wikipedia.
    Menangani teks di dalam link dan elemen yang bisa dilipat (dropdown).
    """
    infobox = soup.select_one("table.infobox")
    data = {}

    if not infobox:
        print("❌ infobox tidak ditemukan")
        return data

    # Hapus elemen yang mengganggu seperti referensi [1], [2], dsb
    for sup in infobox.find_all("sup"):
        sup.decompose()

    rows = infobox.select("tr")
    for row in rows:
        header = row.select_one("th.infobox-label")
        value = row.select_one("td.infobox-data")

        if header and value:
            # Ambil label (kunci)
            key = header.get_text(" ", strip=True)
            key = re.sub(r"\s+", " ", key).replace("•", "").strip()

            # Ambil nilai (value)
            # Menggunakan separator " " agar teks di dalam <a> atau <span> tidak menempel
            val = value.get_text(" ", strip=True)
            val = re.sub(r"\s+", " ", val).strip()

            if val:
                data[key] = val
                
    return data

def extract_all_sections(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return {}

    infobox_data = extract_infobox(soup)
    data = {"INFOBOX": infobox_data}
    
    current_section = "Intro"
    data[current_section] = []

    # Wrapper utama konten Wikipedia
    content_div = soup.select_one("div.mw-parser-output")
    if not content_div:
        return data

    # Tambahkan 'ul' ke dalam pencarian tag
    for tag in content_div.find_all(["h2", "h3", "h4", "p", "ul"]):
        
        # 1. Menentukan Judul Section
        if tag.name in ["h2", "h3", "h4"]:
            title = re.sub(r"\[.*?\]", "", tag.get_text().strip())
            # Skip section yang tidak relevan
            if title.lower() in ["lihat pula", "referensi", "pranala luar", "catatan"]:
                current_section = None
                continue
            current_section = title
            data[current_section] = []

        # Jika current_section None (setelah referensi), lewati
        if not current_section:
            continue

        # 2. Mengambil Paragraf
        if tag.name == "p":
            txt = re.sub(r"\[\d+\]", "", tag.get_text().strip())
            if txt:
                data[current_section].append(txt)

        # 3. Mengambil List (Pahlawan, Daftar Kabupaten, dll)
        elif tag.name == "ul":
            # Periksa apakah ini ul di dalam infobox atau navigasi (kita skip kalau ya)
            if tag.find_parent("table", class_="infobox") or "navbox" in tag.get("class", []):
                continue
                
            items = tag.find_all("li")
            for li in items:
                # Bersihkan teks list
                li_txt = re.sub(r"\[\d+\]", "", li.get_text().strip())
                if li_txt:
                    # Tambahkan bullet point agar rapi di .txt
                    data[current_section].append(f"• {li_txt}")

    return data

def save_to_txt(data, filename):
    """
    Menyimpan hasil ekstraksi ke file TXT dengan format yang rapi.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for section, content in data.items():
            f.write(f"=== {section.upper()} ===\n")
            
            # Jika konten berupa dictionary (khusus INFOBOX)
            if isinstance(content, dict):
                for k, v in content.items():
                    f.write(f"{k}: {v}\n")
            # Jika konten berupa list (paragraf teks)
            else:
                for p in content:
                    f.write(p + "\n\n")
            
            f.write("\n" + "="*40 + "\n\n")

    print(f"✅ Berhasil disimpan ke: {filename}")

def main():
    print(f"🔄 Sedang menyalin data dari: {URL}...")
    
    full_data = extract_all_sections(URL)
    
    if full_data:
        print(f"📊 Section ditemukan: {', '.join(list(full_data.keys()))}")
        save_to_txt(full_data, OUTPUT_FILE)
    else:
        print("❌ Tidak ada data yang berhasil diekstrak.")

if __name__ == "__main__":
    main()