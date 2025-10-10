from datasets import load_dataset
import pandas as pd
import re

#  Load Dataset from Hugging Face 
ds = load_dataset("junaid1993/Dataset_Bot_Detection")
npl_df = ds["train"].to_pandas()

#  2. Standardise column names 
npl_df.columns = npl_df.columns.str.strip().str.lower().str.replace(" ", "_")

#  3. Rename columns if necessary 
if "text_data" not in npl_df.columns:
    for alt in ["text", "tweet", "content"]:
        if alt in npl_df.columns:
            npl_df = npl_df.rename(columns={alt: "text_data"})
            break

if "label" not in npl_df.columns:
    for alt in ["Label", "target", "class", "bot_label"]:
        if alt in npl_df.columns:
            npl_df = npl_df.rename(columns={alt: "label"})
            break

#  4. Drop empty rows 
npl_df = npl_df.dropna(subset=["text_data", "label"]).copy()
npl_df["text_data"] = npl_df["text_data"].astype(str).str.strip()

#  Convert label text → numeric
if npl_df["label"].dtype == object:
    npl_df["label"] = npl_df["label"].astype(str).str.lower().map({"bot": 1, "human": 0})
npl_df["label"] = npl_df["label"].fillna(0).astype(int)

#  Basic text cleaning
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMOJI_RE = re.compile(
    "[" "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]+", flags=re.UNICODE
)

def clean_text(text):
    text = str(text).replace("\n", " ").strip()
    text = URL_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)
    return text.strip()

npl_df["text_clean"] = npl_df["text_data"].apply(clean_text)
npl_df["word_count"] = npl_df["text_clean"].str.split().str.len().fillna(0)

#  Save cleaned dataset
npl_df.to_csv("npl_text_clean.csv", index=False)

print(" Cleaned NLP dataset saved as 'npl_text_clean.csv'")
print("Shape:", npl_df.shape)
print(npl_df.head(3))