import json
import os
import torch
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()
# ------------------------------------
# CONFIG
# ------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "styling-rules2"
DATA_FILE = "outfits_flat_with_gender.json"   # your file

MODEL_NAME = "BAAI/bge-large-en-v1.5"  # 1024-dim embeddings

# ------------------------------------
# INIT CLIENTS
# ------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

print("Loading HuggingFace embedding model...")
model = SentenceTransformer(MODEL_NAME)

# ------------------------------------
# CREATE INDEX IF NOT EXISTS
# ------------------------------------
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating Pinecone index '{INDEX_NAME}' (1024 dimensions)...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# ------------------------------------
# LOAD DATA
# ------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

dataset = load_json(DATA_FILE)
print(f"Loaded {len(dataset)} outfit entries.")


# ------------------------------------
# EMBEDDING TEXT BUILDER (ADAPTED)
# ------------------------------------
def build_embedding_text(item):
    """
    Your ORIGINAL format — but adapted to outfit fields.
    No changes in structure at all.
    """

    metadata = item.get("metadata", {})
    raw = item.get("raw", {})

    top = metadata.get("top", "")
    bottom = metadata.get("bottom", "")
    shoes = ", ".join(metadata.get("shoes", []) or [])
    purses = ", ".join(metadata.get("purses", []) or [])
    category = metadata.get("category", "")
    style = metadata.get("style", "")
    seasons = ", ".join(metadata.get("season", []) or [])
    occasions = ", ".join(metadata.get("occasions", []) or [])

    style_notes = raw.get("style_notes", "")
    color_tips = ", ".join(raw.get("color_matching_tips", []) or [])

    original_text = item.get("text", "")

    combined = f"""
    OUTFIT:
    {original_text}

    DESCRIPTION:
    {style_notes}

    STYLE: {style}
    CATEGORY: {category}
    TOP: {top}
    BOTTOM: {bottom}
    SHOES: {shoes}
    PURSES: {purses}

    COLOR TIPS: {color_tips}
    OCCASIONS: {occasions}
    SEASONS: {seasons}
    """

    return combined.strip()


# ------------------------------------
# EMBEDDING FUNCTION (UNCHANGED)
# ------------------------------------
def embed(text):
    vec = model.encode(text, convert_to_tensor=True)
    return vec.cpu().tolist()


# ------------------------------------
# UPSERT INTO PINECONE (UNCHANGED)
# ------------------------------------
batch = []
batch_size = 50

print("Uploading vectors to Pinecone...")

for item in tqdm(dataset):

    embed_input = build_embedding_text(item)
    vector = embed(embed_input)

    # metadata EXACT same structure as your example
    record = {
        "id": item["id"],
        "values": vector,
        "metadata": {
            "text": item["text"],
            "style": item["metadata"].get("style", ""),
            "category": item["metadata"].get("category", ""),
            "season": item["metadata"].get("season", []),
            "occasions": item["metadata"].get("occasions", []),
            "top": item["metadata"].get("top", ""),
            "bottom": item["metadata"].get("bottom", ""),
            "shoes": item["metadata"].get("shoes", []),
            "gender": item["metadata"].get("gender", []),
            "source_id": item["id"]
        }
    }

    # Clean metadata to avoid null errors
    for key, value in record["metadata"].items():
        if value is None:
            record["metadata"][key] = ""
        if isinstance(value, list):
            record["metadata"][key] = [str(x) if x is not None else "" for x in value]

    batch.append(record)

    if len(batch) >= batch_size:
        index.upsert(vectors=batch)
        batch = []

if batch:
    index.upsert(vectors=batch)

print("\n✨ Upload complete — outfits stored in Pinecone!")
print("Index:", INDEX_NAME)
