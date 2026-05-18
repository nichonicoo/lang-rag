import pandas as pd
import json

# load
df = pd.read_json("toba_qwen_sectioned.jsonl", lines=True)

# ambil user + assistant jadi satu string
def flatten(messages):
    user = next((m['content'] for m in messages if m['role']=='user'), "")
    assistant = next((m['content'] for m in messages if m['role']=='assistant'), "")
    return user.strip() + " || " + assistant.strip()

df['text'] = df['messages'].apply(flatten)

# detect duplicate
duplicates = df[df.duplicated('text', keep=False)]

print("Total duplicates:", len(duplicates))
duplicates.head()

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

similar_pairs = []

threshold = 0.9  # bisa adjust

for i in range(len(embeddings)):
    sims = cosine_similarity([embeddings[i]], embeddings)[0]
    
    for j in range(i+1, len(sims)):
        if sims[j] > threshold:
            similar_pairs.append((i, j, sims[j]))

print("Near duplicates found:", len(similar_pairs))