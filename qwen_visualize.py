import streamlit as st
import pandas as pd

df = pd.read_json("toba_qwen_sectioned.jsonl", lines=True)

st.title("Dataset Viewer")

idx = st.slider("Index", 0, len(df)-1, 0)

for m in df.iloc[idx]['messages']:
    st.write(f"**{m['role']}**: {m['content']}")