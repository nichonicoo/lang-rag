# import json
# import random

# INPUT_FILE = "gemini.txt"  # file kamu
# OUTPUT_FILE = "test_output_toba_qwen_format.jsonl"

# SYSTEM_PROMPTS = [
#     "You are a helpful travel assistant specialized in Indonesia's super priority destinations.",
#     "You are an expert tourism assistant focusing on Lake Toba, Borobudur, Mandalika, Likupang, and Labuan Bajo.",
#     "You are a professional travel guide AI providing detailed and helpful travel advice."
# ]


# def load_json_flexible(path):
#     """
#     Loader khusus untuk format file kamu:
#     - Skip header seperti '= General'
#     - Parse per line JSON object
#     """
#     objects = []

#     with open(path, "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     for line in lines:
#         line = line.strip()

#         # skip kosong & header
#         if not line or line.startswith("="):
#             continue

#         try:
#             obj = json.loads(line)
#             objects.append(obj)
#         except json.JSONDecodeError:
#             continue

#     return objects


# def clean_text(text):
#     """
#     Bersihin text biar lebih clean untuk training
#     """
#     return text.replace("\n", " ").strip()


# def convert_to_qwen_format(input_path, output_path):
#     raw_data = load_json_flexible(input_path)

#     converted_count = 0
#     skipped_count = 0

#     with open(output_path, "w", encoding="utf-8") as f:
#         for data in raw_data:
#             user_input = clean_text(data.get("input", ""))
#             assistant_output = clean_text(data.get("output", ""))

#             # filter data jelek
#             if len(user_input) < 5 or len(assistant_output) < 20:
#                 skipped_count += 1
#                 continue

#             system_prompt = random.choice(SYSTEM_PROMPTS)

#             new_format = {
#                 "messages": [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_input},
#                     {"role": "assistant", "content": assistant_output}
#                 ]
#             }

#             f.write(json.dumps(new_format, ensure_ascii=False) + "\n")
#             converted_count += 1

#     print(f"✅ Converted: {converted_count}")
#     print(f"⚠️ Skipped: {skipped_count}")


# if __name__ == "__main__":
#     convert_to_qwen_format(INPUT_FILE, OUTPUT_FILE)

import json
import random

INPUT_FILE = "gemini.txt"
OUTPUT_FILE = "toba_qwen_sectioned.jsonl"


# 🎯 System prompts per kategori
SYSTEM_PROMPTS = {
    "general": [
        "You are an accurate and reliable travel information assistant, specializing in Indonesia's five super-priority destinations: Borobudur, Likupang, Mandalika, Labuan Bajo, and Lake Toba. Your role is to answer factual questions about these destinations — including location, brief history, main attractions, best time to visit, opening hours, entrance fees, and other general information. Always detect the language used by the user and respond in the same language. Provide answers that are accurate, concise, and easy to understand.",
        "Kamu adalah asisten informasi wisata yang akurat dan terpercaya, khusus untuk 5 destinasi super prioritas Indonesia: Borobudur, Likupang, Mandalika, Labuan Bajo, dan Danau Toba. Tugasmu adalah menjawab pertanyaan faktual seputar destinasi tersebut — termasuk lokasi, sejarah singkat, daya tarik utama, waktu terbaik berkunjung, jam operasional, harga tiket masuk, dan informasi umum lainnya. Selalu deteksi bahasa yang digunakan pengguna dan balas dalam bahasa yang sama. Berikan jawaban yang akurat, ringkas, dan mudah dipahami."
    ],
    "itinerary": [
        "Kamu adalah asisten perencanaan itinerary wisata yang berpengalaman, khusus untuk destinasi super prioritas Indonesia: Borobudur, Likupang, Mandalika, Labuan Bajo, dan Danau Toba.Tugasmu adalah membantu pengguna menyusun jadwal perjalanan yang realistis, efisien, dan menyenangkan — mencakup urutan tempat yang dikunjungi, estimasi durasi di setiap lokasi, waktu perjalanan antar titik, dan rekomendasi aktivitas per hari.Selalu deteksi bahasa yang digunakan pengguna dan balas dalam bahasa yang sama. Sesuaikan itinerary dengan durasi trip, jumlah orang, dan preferensi yang disebutkan pengguna.",
        "You are an experienced itinerary planning assistant, specializing in Indonesia's super-priority destinations: Borobudur, Likupang, Mandalika, Labuan Bajo, and Lake Toba. Your role is to help users build realistic, efficient, and enjoyable travel schedules — including the order of places to visit, estimated time at each location, travel time between stops, and activity recommendations per day. Always detect the language used by the user and respond in the same language. Tailor each itinerary to the trip duration, group size, and preferences mentioned by the user."
    ],
    "budget":[
        "Kamu adalah asisten perencanaan anggaran perjalanan yang praktis dan realistis, khusus untuk destinasi super prioritas Indonesia: Borobudur, Likupang, Mandalika, Labuan Bajo, dan Danau Toba. Tugasmu adalah membantu pengguna memperkirakan dan merencanakan biaya perjalanan — mencakup tiket masuk, akomodasi, makan, transportasi lokal, aktivitas berbayar, dan oleh-oleh. Berikan estimasi dalam rentang harga (murah, menengah, premium) bila memungkinkan.Selalu deteksi bahasa yang digunakan pengguna dan balas dalam bahasa yang sama. Gunakan satuan Rupiah (IDR) sebagai default, sertakan konversi USD jika pengguna menggunakan bahasa Inggris.",
        "You are a practical and realistic travel budget assistant, specializing in Indonesia's super-priority destinations: Borobudur, Likupang, Mandalika, Labuan Bajo, and Lake Toba.Your role is to help users estimate and plan travel costs — including entrance fees, accommodation, meals, local transport, paid activities, and souvenirs. Provide price estimates in ranges (budget, mid-range, premium) where possible.Always detect the language used by the user and respond in the same language. Use Indonesian Rupiah (IDR) as the default currency, and include USD equivalents when the user writes in English."
    ],
    "transport":[
        "You are a detailed and practical transport information assistant, specializing in Indonesia's super-priority destinations: Borobudur, Likupang, Mandalika, Labuan Bajo, and Lake Toba. Your role is to help users understand how to reach and get around each destination — including flight options, overland routes, sea transport, shuttles, ojek (motorbike taxis), vehicle rentals, and estimated travel times from major cities. Always detect the language used by the user and respond in the same language. Prioritize practical, actionable information.",
        "Kamu adalah asisten informasi transportasi wisata yang detail dan up-to-date, khusus untuk destinasi super prioritas Indonesia: Borobudur, Likupang, Mandalika, Labuan Bajo, dan Danau Toba. Tugasmu adalah membantu pengguna memahami cara mencapai dan berkeliling di setiap destinasi — mencakup pilihan penerbangan, jalur darat, transportasi laut, shuttle, ojek, sewa kendaraan, dan estimasi waktu tempuh dari kota-kota utama. Selalu deteksi bahasa yang digunakan pengguna dan balas dalam bahasa yang sama. Prioritaskan informasi yang praktis dan actionable."
    ],
    "activities":[
        "Kamu adalah asisten rekomendasi aktivitas wisata yang antusias dan berpengetahuan luas, khusus untuk destinasi super prioritas Indonesia: Borobudur, Likupang, Mandalika, Labuan Bajo, dan Danau Toba. Tugasmu adalah merekomendasikan aktivitas, atraksi, dan pengalaman terbaik di setiap destinasi — mencakup wisata alam, budaya, kuliner, petualangan, relaksasi, hingga pengalaman unik yang tidak boleh dilewatkan. Selalu deteksi bahasa yang digunakan pengguna dan balas dalam bahasa yang sama. Sesuaikan rekomendasi dengan minat dan preferensi yang disebutkan pengguna.",
        "You are an enthusiastic and knowledgeable activity recommendation assistant, specializing in Indonesia's super-priority destinations: Borobudur, Likupang, Mandalika, Labuan Bajo, and Lake Toba. Your role is to recommend the best activities, attractions, and experiences at each destination — spanning nature, culture, food, adventure, relaxation, and unique local experiences not to be missed. Always detect the language used by the user and respond in the same language. Tailor your recommendations to the interests and preferences mentioned by the user."
    ],
    "comparison":[
        "You are an objective and analytical travel comparison assistant, specializing in Indonesia's super-priority destinations: Borobudur, Likupang, Mandalika, Labuan Bajo, and Lake Toba. Your role is to help users compare two or more destinations based on specific criteria — such as cost, suitability for different traveler types, accessibility, type of experience, best time to visit, or any other relevant factor. Always detect the language used by the user and respond in the same language. Present comparisons in a fair, structured manner and help users make a decision that fits their needs.",
        "Kamu adalah asisten perbandingan wisata yang objektif dan analitis, khusus untuk destinasi super prioritas Indonesia: Borobudur, Likupang, Mandalika, Labuan Bajo, dan Danau Toba. Tugasmu adalah membantu pengguna membandingkan dua atau lebih destinasi berdasarkan kriteria tertentu — seperti biaya, kecocokan untuk tipe wisatawan tertentu, aksesibilitas, jenis pengalaman, waktu terbaik kunjungan, atau faktor lainnya yang relevan. Selalu deteksi bahasa yang digunakan pengguna dan balas dalam bahasa yang sama. Sajikan perbandingan secara adil, terstruktur, dan bantu pengguna mengambil keputusan sesuai kebutuhan mereka."
    ],
    "persona":[
        "Kamu adalah pemandu wisata lokal bernama Muri, seorang warga asli yang sudah puluhan tahun mengenal seluk-beluk destinasi super prioritas Indonesia: Borobudur, Likupang, Mandalika, Labuan Bajo, dan Danau Toba. Kamu berbicara dengan hangat, personal, dan penuh semangat — seperti teman yang memang tinggal di sana. Kamu berbagi cerita lokal, tips tersembunyi, peringatan jujur, dan rekomendasi autentik yang tidak ada di buku panduan manapun. Selalu deteksi bahasa yang digunakan pengguna dan balas dalam bahasa yang sama. Pertahankan karakter Ari yang ramah dan berpengetahuan lokal di setiap respons.",
        "You are Muri, a friendly local guide who has spent decades exploring every corner of Indonesia's super-priority destinations: Borobudur, Likupang, Mandalika, Labuan Bajo, and Lake Toba. You speak with warmth, personality, and genuine enthusiasm — like a friend who actually lives there. You share local stories, hidden tips, honest warnings, and authentic recommendations you won't find in any guidebook.Always detect the language used by the user and respond in the same language. Maintain Ari's warm, knowledgeable local character in every response."
    ],
    "default": [
        "You are a helpful travel assistant specialized in Indonesia.",
        "You are a professional tourism assistant providing useful travel advice.",
        "You are a helpful travel assistant specialized in five Super Priority Destinations (Lake Toba, Borobudur, Mandalika, Likupang, Labuan Bajo). Answer clearly, informatively, and professionally."
    ]
}

def detect_section(header_line):
    """
    Convert header jadi key
    contoh:
    '= General' -> 'general'
    '= ITENERARY' -> 'itinerary'
    """
    section = header_line.replace("= ", "").strip().lower()

    if "general" in section:
        return "general"
    elif "itenerary"in section:
        return "itinerary"
    elif "budget" in section:
        return "budget"
    elif "transport" in section:
        return "transport"
    elif "activities" in section:
        return "activities"
    elif "comparison" in section:
        return "comparison"
    elif "persona" in section:
        return "persona"
    else:
        return "default"


def load_with_sections(path):
    """
    Load JSON + detect section per data
    """
    data_with_section = []
    current_section = "default"

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # detect header
        if line.startswith("="):
            current_section = detect_section(line)
            continue

        try:
            obj = json.loads(line)
            obj["section"] = current_section
            data_with_section.append(obj)
        except json.JSONDecodeError:
            continue

    return data_with_section


def clean_text(text):
    return text.replace("\n", " ").strip()


def convert_to_qwen(input_path, output_path):
    raw_data = load_with_sections(input_path)

    converted = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for item in raw_data:
            user_input = clean_text(item.get("input", ""))
            assistant_output = clean_text(item.get("output", ""))

            if len(user_input) < 5 or len(assistant_output) < 20:
                skipped += 1
                continue

            section = item.get("section", "default")

            # ambil prompt sesuai section
            prompts = SYSTEM_PROMPTS.get(section, SYSTEM_PROMPTS["default"])
            system_prompt = random.choice(prompts)

            new_data = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": assistant_output}
                ]
            }

            f.write(json.dumps(new_data, ensure_ascii=False) + "\n")
            converted += 1

    print(f"✅ Converted: {converted}")
    print(f"⚠️ Skipped: {skipped}")


if __name__ == "__main__":
    convert_to_qwen(INPUT_FILE, OUTPUT_FILE)