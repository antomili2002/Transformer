import json
import os
import re
from datasets import load_dataset
from tqdm import tqdm

os.makedirs("data", exist_ok=True)

dataset = load_dataset("wmt17", "de-en")

def clean_data(data, min_length=5, max_length=64):
    whitelist = "abcdefghijklmnopqrstuvwxyzÄÖÜäöüßABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥"
    url_pattern = re.compile(r'http\S+|www\S+|https\S+')
    html_pattern = re.compile(r'<.*?>')

    cleaned_data = []
    for item in tqdm(data):
        de = item['translation']['de']
        en = item['translation']['en']

        # Remove URLs and HTML tags
        de = re.sub(url_pattern, '', de)
        de = re.sub(html_pattern, '', de)
        en = re.sub(url_pattern, '', en)
        en = re.sub(html_pattern, '', en)

        # Keep only whitelisted characters
        de = re.sub(f"[^{re.escape(whitelist)}]", " ", de)
        en = re.sub(f"[^{re.escape(whitelist)}]", " ", en)

        # Normalize spaces
        #de = re.sub(r'\s+', ' ', de).strip()
        #en = re.sub(r'\s+', ' ', en).strip()

        # Skip empty samples
        if not de or not en:
            continue

        # Filter by length
        de_len = len(de.split())
        en_len = len(en.split())
        if not (min_length <= de_len <= max_length and min_length <= en_len <= max_length):
            continue

        # Filter by ratio (0.67 ≤ ratio ≤ 1.5)
        ratio = de_len / en_len if en_len > 0 else 0
        if 0.67 <= ratio <= 1.5:
            cleaned_data.append({'src': de, 'tgt': en})

    return cleaned_data

print("Cleaning training data...")
cleaned_train_dataset = clean_data(dataset['train'])

print("Cleaning test data...")
cleaned_test_dataset = clean_data(dataset['test'])

print("Cleaning validation data...")
cleaned_validation_dataset = clean_data(dataset['validation'])

print("Preparing texts for tokenizer training...")
train_texts = [item['translation']['de'] + " " + item['translation']['en'] for item in dataset['train']]

with open("data/cleaned_wmt17_de_en_split_train.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_train_dataset, f, ensure_ascii=False, indent=2)

with open("data/cleaned_wmt17_de_en_split_validation.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_validation_dataset, f, ensure_ascii=False, indent=2)

with open("data/cleaned_wmt17_de_en_split_test.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_test_dataset, f, ensure_ascii=False, indent=2)

train_texts = [ex["src"] for ex in cleaned_train_dataset] + [
    ex["tgt"] for ex in cleaned_train_dataset
]

with open("data/cleaned_wmt17_de_en_texts_for_tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(train_texts, f, ensure_ascii=False, indent=2)

print("Done. Saved cleaned splits and tokenizer texts in ./data")