from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import json

pc = Pinecone(api_key="")
pinecone_index = pc.Index("styling-rules2")
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def build_query_text(preferences):
    """Convert user preferences into a meaningful semantic query."""
    query_parts = []

    gender = preferences.get("gender")
    if gender:
        query_parts.append(f"Styling for a {gender} user")

    age_range = preferences.get("age_range")
    if age_range:
        query_parts.append(f"Age range: {age_range}")

    style_goals = preferences.get("style_goal")
    if style_goals:
        query_parts.append(f"Style goals: {', '.join(style_goals)}")

    difficult = preferences.get("difficult_occasion")
    if difficult:
        query_parts.append(f"Outfits for difficult occasions: {', '.join(difficult)}")

    boldness = preferences.get("outfit_boldness")
    if boldness:
        query_parts.append(f"Boldness preference: {boldness}")

    colors = preferences.get("color_comfort")
    if colors:
        query_parts.append(f"Color comfort: {colors}")

    fit = preferences.get("preferred_fit")
    if fit:
        query_parts.append(f"Preferred fit: {fit}")

    style_pref = preferences.get("style_preference")
    if style_pref:
        query_parts.append(f"Style preference: {', '.join(style_pref)}")

    # Final merged query text
    return "\n".join(query_parts).strip()

def search_styling_rules(preferences, pinecone_index, top_k=5):
    """Query Pinecone using preferences to get styling rule chunks."""
    top_k = int(top_k)  # 🔥 ensures top_k is always an integer

    if top_k < 1:
        top_k = 1
        
    query_text = build_query_text(preferences)
    query_vector = embedding_model.encode(query_text).tolist()

    # 🔥 Now pinecone_index is an actual Index object
    response = pinecone_index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    # Combine metadata text content
    retrieved_chunks = [
        match.metadata.get("text", "")
        for match in response.matches
    ]

    return "\n\n".join(retrieved_chunks)
