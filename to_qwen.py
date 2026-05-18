import json

INPUT_FILE = "borobudur_toba_500.json"
OUTPUT_FILE = "borobudur_toba_500_qwen_format.jsonl"

SYSTEM_PROMPT = "You are a helpful travel assistant specialized in five "Super Priority Destinations" (Lake Toba, Borobudur, Mandalika, Likupang, Labuan Bajo)"


def load_json_flexible(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # CASE 1: proper JSON array
    if content.startswith("["):
        return json.loads(content)

    # CASE 2: JSON objects dipisah newline / manual
    objects = []
    buffer = ""

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        buffer += line

        if line.endswith("}"):
            try:
                obj = json.loads(buffer)
                objects.append(obj)
                buffer = ""
            except:
                continue

    return objects


def convert_to_qwen_format(input_path, output_path):
    raw_data = load_json_flexible(input_path)

    converted_data = []

    for data in raw_data:
        user_input = data.get("input", "").strip()
        assistant_output = data.get("output", "").strip()

        if not user_input or not assistant_output:
            continue

        new_format = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_output}
            ]
        }

        converted_data.append(new_format)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Converted {len(converted_data)} data to Qwen format")


if __name__ == "__main__":
    convert_to_qwen_format(INPUT_FILE, OUTPUT_FILE)