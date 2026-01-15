import json
import torch
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# ------------------------------------
# CONFIG
# ------------------------------------
PINECONE_API_KEY = ""
INDEX_NAME = "styling-rules2"
DATA_FILE = "preprocessed_fashion_chunks_final_ready.json" 

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
print(f"Loaded {len(dataset)} styling rule entries.")

# ------------------------------------
# EMBEDDING TEXT BUILDER
# ------------------------------------
def build_embedding_text(item):
    """
    Combine multiple fields into one semantic-rich embedding input.
    This makes the vector DB much more accurate.
    """

    style_types = ", ".join(item.get("style_type", []))
    techniques = ", ".join(item.get("technique", []))
    seasons = ", ".join(item.get("season", []))
    occasions = ", ".join(item.get("occasion", []))
    summary = item.get("summary", "")
    text = item.get("text", "")

    combined = f"""
    STYLING RULE:
    {text}

    SUMMARY:
    {summary}

    STYLE TYPES: {style_types}
    TECHNIQUES: {techniques}
    OCCASIONS: {occasions}
    SEASONS: {seasons}
    """

    return combined.strip()

# ------------------------------------
# EMBEDDING FUNCTION
# ------------------------------------
def embed(text):
    vec = model.encode(text, convert_to_tensor=True)
    return vec.cpu().tolist()

# ------------------------------------
# UPSERT INTO PINECONE
# ------------------------------------
batch = []
batch_size = 50

print("Uploading vectors to Pinecone...")

for item in tqdm(dataset):

    embed_input = build_embedding_text(item)
    vector = embed(embed_input)

    record = {
        "id": item["id"],
        "values": vector,
        "metadata": {
            "text": item["text"],
            "summary": item.get("summary", ""),
            "style_type": item.get("style_type", []),
            "technique": item.get("technique", []),
            "season": item.get("season", []),
            "occasion": item.get("occasion", []),
            "source_id": item["id"]
        }
    }

    batch.append(record)

    if len(batch) >= batch_size:
        index.upsert(vectors=batch)
        batch = []

if batch:
    index.upsert(vectors=batch)

print("\n✨ Upload complete — improved styling rules stored in Pinecone!")
print("Index:", INDEX_NAME)
