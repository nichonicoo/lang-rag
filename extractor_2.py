import requests
import re
from bs4 import BeautifulSoup

URL = "https://id.wikipedia.org/wiki/Borobudur?action=render"
OUTPUT_FILE = "borobudur_full.txt"

def extract_infobox(soup):
    """Mengekstraksi data infobox (diperbarui dari jawaban sebelumnya)."""
    infobox = soup.select_one("table.infobox")
    data = {}
    if not infobox: return data

    for sup in infobox.find_all("sup"): sup.decompose()

    rows = infobox.select("tr")
    for row in rows:
        header = row.select_one("th.infobox-label")
        value = row.select_one("td.infobox-data")
        if header and value:
            key = header.get_text(" ", strip=True)
            key = re.sub(r"\s+", " ", key).replace("•", "").strip()
            val = value.get_text(" ", strip=True)
            val = re.sub(r"\s+", " ", val).strip()
            if val: data[key] = val
    return data

def table_to_markdown(table_tag):
    """Mengonversi wikitable HTML menjadi tabel Markdown."""
    rows = table_tag.find_all('tr')
    md_rows = []
    
    for i, row in enumerate(rows):
        cells = row.find_all(['th', 'td'])
        clean_cells = []
        for cell in cells:
            for img in cell.find_all('img'): img.decompose()
            for sup in cell.find_all('sup'): sup.decompose()
            txt = cell.get_text(" ", strip=True).replace("\n", " ")
            clean_cells.append(txt)
        
        if not clean_cells: continue
            
        md_rows.append("| " + " | ".join(clean_cells) + " |")
        if i == 0:
            md_rows.append("| " + " | ".join(["---"] * len(clean_cells)) + " |")
            
    return "\n".join(md_rows)

def get_image_marker(img_tag):
    """
    Mengambil URL gambar, melengkapinya dengan https:, 
    dan mengembalikannya dalam format penanda Markdown: ![caption](url)
    """
    if not img_tag:
        return ""

    # 1. Ambil URL Gambar dari atribut 'src'
    img_url = img_tag.get('src')
    if not img_url:
        return ""

    # 2. Perbaiki URL yang relatif (Wikipedia sering pakai //upload...)
    if img_url.startswith('//'):
        img_url = 'https:' + img_url
    
    # 3. Ambil Keterangan Gambar (Alt Text atau Title)
    # Wikipedia biasanya menaruh keterangan di atribut 'alt' pada tag <img>
    # atau di dalam tag <div class="thumbcaption"> di sekitar gambar.
    caption = img_tag.get('alt', 'Gambar Wikipedia')
    
    # Temukan caption yang lebih baik jika gambar ada di dalam .thumb
    parent_thumb = img_tag.find_parent('div', class_='thumb')
    if parent_thumb:
        caption_div = parent_thumb.find('div', class_='thumbcaption')
        if caption_div:
            # Ambil teks caption dan bersihkan dari teks [sunting]
            caption = caption_div.get_text(" ", strip=True)
            caption = re.sub(r"\s+", " ", caption).strip()

    # 4. Buat penanda Markdown
    return f"\n\n![{caption}]({img_url})\n\n"

def extract_all_sections(url):
    """
    Mengekstraksi infobox, teks (p, ul, table), dan menyisipkan penanda gambar.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
    except Exception as e:
        print(f"❌ Gagal mengambil URL: {e}")
        return {}

    data = {"INFOBOX": extract_infobox(soup)}
    
    # Wrapper utama konten Wikipedia
    content_div = soup.select_one("div.mw-parser-output")
    if not content_div: return data

    current_section = "Intro"
    data[current_section] = []

    # Iterasi elemen dalam wrapper utama
    for tag in content_div.find_all(["h2", "h3", "h4", "p", "ul", "table", "div"]):
        
        # 1. Judul Section
        if tag.name in ["h2", "h3", "h4"]:
            title = re.sub(r"\[.*?\]", "", tag.get_text().strip())
            if title.lower() in ["lihat pula", "referensi", "pranala luar", "catatan"]:
                current_section = None
                continue
            current_section = title
            data[current_section] = []

        if not current_section: continue

        # 2. Gambar (Biasanya di dalam div.thumb atau div.mw-file-description)
        # Penanganan khusus agar gambar muncul SEBELUM paragraf teksnya.
        if tag.name == "div" and ("thumb" in tag.get("class", []) or "mw-file-description" in tag.get("class", [])):
            img_tag = tag.find('img')
            if img_tag:
                marker = get_image_marker(img_tag)
                if marker:
                    data[current_section].append(marker)

        # 3. Paragraf Teks
        elif tag.name == "p":
            txt = re.sub(r"\[\d+\]", "", tag.get_text().strip())
            if txt:
                data[current_section].append(txt)

        # 4. List (ul)
        elif tag.name == "ul":
            # Skip list navigasi
            if tag.find_parent("table", class_="infobox") or "navbox" in tag.get("class", []):
                continue
            for li in tag.find_all("li"):
                li_txt = re.sub(r"\[\d+\]", "", li.get_text().strip())
                if li_txt:
                    data[current_section].append(f"• {li_txt}")

        # 5. Tabel Teks (wikitable)
        elif tag.name == "table" and "wikitable" in tag.get("class", []):
            md_table = table_to_markdown(tag)
            if md_table:
                data[current_section].append(f"\n\n{md_table}\n\n")

    return data

def save_to_txt(data, filename):
    """Menyimpan hasil ekstraksi ke file TXT."""
    with open(filename, "w", encoding="utf-8") as f:
        for section, content in data.items():
            f.write(f"=== {section.upper()} ===\n")
            
            if isinstance(content, dict): # INFOBOX
                for k, v in content.items():
                    f.write(f"{k}: {v}\n")
            else: # List Teks/Gambar
                for item in content:
                    f.write(item + "\n\n")
            
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