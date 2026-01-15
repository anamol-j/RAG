from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
import json
from vector_search import search_styling_rules

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.4,
    groq_api_key=os.getenv("GROQ_API_KEY","")
)

styling_prompt = PromptTemplate(
    input_variables=["preferences", "styling_rules"],
    template ="""
You are a professional personal stylist with deep knowledge of outfit coordination, color theory, proportions, and modern fashion trends.

Below is the USER'S PREFERENCE PROFILE.
Some fields may be missing — do NOT invent anything.
Always adapt outfits based on the user's stated gender.

--- USER PREFERENCES ---
{preferences}

Below are the TOP RETRIEVED STYLING RULES from the vector store.
These are factual styling guidelines you must use as reference.
Do NOT copy them word-for-word; interpret them and apply them creatively.

--- STYLING RULES ---
{styling_rules}
----------------------

TASK:
Generate a set of personalized outfit recommendations **one by one**, using:
1. The user's personal preferences
2. The retrieved styling rules
3. Your own fashion expertise

RULES:
- Every outfit MUST match the user's gender.
- Apply the user's fit preference, color comfort, outfit boldness, and style goals.
- If the user listed difficult occasions, include outfits tailored for them.
- Use the retrieved styling rules to guide layering, mixing proportions, textures, and color logic.
- Keep outfits realistic, modern, and wearable.
- Include a short explanation for each outfit.
- Do NOT mention the vector store or the chunks.

FORMAT:
1. Outfit #1  
   - Description  
   - Why it fits  
2. Outfit #2  
   - Description  
   - Why it fits  
3. Outfit #3  
   - Description  
   - Why it fits  

Generate 3–5 outfits depending on the richness of preferences.
Output ONLY the outfits and explanations.
"""
)

def generate_styling_suggestions(preferences, pinecone_index):
    # 1. Get styling rules from Pinecone
    styling_rules = search_styling_rules(preferences, pinecone_index)

    # 2. Send everything to the model
    prompt_text = styling_prompt.format(
        preferences=json.dumps(preferences, indent=2),
        styling_rules=styling_rules
    )

    response = llm.invoke(prompt_text)
    return response.content
