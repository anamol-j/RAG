import streamlit as st
from question_bank import get_random_questions
from style_summary import generate_styling_suggestions
from profile_utils import normalize_preferences
from pinecone import Pinecone

pc = Pinecone(api_key="")
pinecone_index = pc.Index("styling-rules")

st.set_page_config(page_title="Style Preferences", layout="centered")
st.title("Personal Style Questionnaire")

# -----------------------------------
# LOAD RANDOM QUESTIONS ONCE
# -----------------------------------
if "questions" not in st.session_state:
    st.session_state.questions = get_random_questions(total=6)

questions = st.session_state.questions
responses = {}

# -----------------------------------
# RENDER QUESTIONS
# -----------------------------------
for q in questions:
    if q["type"] == "radio":
        responses[q["id"]] = st.radio(q["label"], q["options"], key=q["id"])

    elif q["type"] == "multiselect":
        responses[q["id"]] = st.multiselect(q["label"], q["options"], key=q["id"])

    elif q["type"] == "text":
        responses[q["id"]] = st.text_input(q["label"], key=q["id"])

# -----------------------------------
# SUBMIT BUTTON
# -----------------------------------
if st.button("Submit"):
    # 1️⃣ Normalize the response so missing fields don’t break the LLM
    normalized = normalize_preferences(responses)

    st.subheader("Normalized Profile (clean JSON)")
    st.json(normalized)

    # 2️⃣ Send normalized profile to the LLM
    with st.spinner("Generating style summary..."):
        summary = generate_styling_suggestions(normalized, pinecone_index)

    st.subheader("Generated Style Summary")
    st.write(summary)
