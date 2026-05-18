import json

INPUT_FILE = "converted_500_winsen.json"
OUTPUT_FILE = "dataset_500_winsen_qwen_format.jsonl"

SYSTEM_PROMPT = (
    "You are a helpful travel assistant specialized in five Super Priority Destinations "
    "(Lake Toba, Borobudur, Mandalika, Likupang, Labuan Bajo). "
    "Answer clearly, informatively, and professionally."
)


def load_json_flexible(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # CASE 1: JSON array
    try:
        if content.startswith("["):
            return json.loads(content)
    except json.JSONDecodeError:
        pass

    # CASE 2: JSON Lines (JSONL)
    objects = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            objects.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # CASE 3: fallback (manual buffer)
    if not objects:
        buffer = ""
        for line in content.splitlines():
            buffer += line.strip()
            if line.strip().endswith("}"):
                try:
                    objects.append(json.loads(buffer))
                    buffer = ""
                except:
                    continue

    return objects


def clean_text(text):
    return text.replace("\n", " ").strip()


def convert_to_qwen_format(input_path, output_path):
    raw_data = load_json_flexible(input_path)

    converted_count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for data in raw_data:
            user_input = clean_text(data.get("input", ""))
            assistant_output = clean_text(data.get("output", ""))

            if not user_input or not assistant_output:
                continue

            new_format = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": assistant_output}
                ]
            }

            f.write(json.dumps(new_format, ensure_ascii=False) + "\n")
            converted_count += 1

    print(f"✅ Converted {converted_count} data to Qwen format")


if __name__ == "__main__":
    convert_to_qwen_format(INPUT_FILE, OUTPUT_FILE)