import streamlit as st
import sys

sys.path.extend(["source/llm_processing"])
sys.path.extend(["source/ai_generated_detection"])

from llm_processing import generate_n_essays
from ai_generated_detection import predict_ai_generated

# Constant
number_of_essays_generated = 10
top_choosing_essay = 1

def get_best_essay(prompt, num_essays, grod_api_key, top_k=1):
    # Step 1: Generate essays
    essays = generate_n_essays(prompt, num_essays, grod_api_key)

    # Step 2: Get AI-generated probability for each essay
    scored_essays = [(essay, predict_ai_generated(essay)) for essay in essays]

    # Step 3: Sort essays by AI probability (ascending)
    scored_essays.sort(key=lambda x: x[1])

    # Step 4: Select top-k essays with the lowest AI probability
    best_essays = scored_essays[:top_k]

    return best_essays 

# Main UI
st.title("🔗 Easy essay")
st.caption("Auto generate short essay (less than 300 words) for your input topic!!!")

with st.sidebar:
    grod_api_key = st.text_input("Grod API Key", type="password")
    "[Get an Grod API key](https://console.groq.com/keys)"

with st.form("my_form"):
    text = st.text_area(
        "Enter your topic:",
        "Example: Many things that used to be done in home by hands are now being done by machines. Does the development bring more advantages or disadvantages?",
    )
    submitted = st.form_submit_button("Submit")
    if not grod_api_key.startswith("gsk_"):
        st.warning("Please enter your Grod API key!", icon="⚠")
    if submitted and grod_api_key.startswith("gsk_"):
        best_essays = get_best_essay(text, number_of_essays_generated, grod_api_key, top_choosing_essay)
        for essay, prob in best_essays:
            st.info(essay)
            st.info(f"AI detected probability: {prob}")