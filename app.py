import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Fetch API keys from environment
spoonacular_api_key = os.getenv("SPOONACULAR_API_KEY")
hf_api_key = os.getenv("HF_API_KEY")


# Streamlit App Title
st.title("Dynamic Recipe Generator 🍴")

# Input Section
st.sidebar.header("Your Ingredients and Preferences")
ingredients = st.sidebar.text_area("Enter your ingredients (comma-separated):", placeholder="e.g., tomatoes, chicken, garlic")
dietary_pref = st.sidebar.selectbox("Dietary Preference:", ["None", "Vegan", "Vegetarian", "Gluten-Free"])



#ead0cc785365459c9bb4b9f9832e15ad

import requests

# Function to fetch recipes 
def fetch_recipes(ingredients, dietary_pref):
    api_key = spoonacular_api_key
    base_url = "https://api.spoonacular.com/recipes/findByIngredients"

    # API parameters
    params = {
        "ingredients": ingredients,
        "number": 3,  # Fetch top 3 recipes
        "apiKey": api_key,
        "diet": dietary_pref.lower() if dietary_pref != "None" else None
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch recipes. Please try again.")
        return []

# Display recipes
if st.sidebar.button("Generate Recipe", key="generate_recipe"):
    st.write("Generating recipe...")
    if ingredients:
        recipes = fetch_recipes(ingredients, dietary_pref)
        for recipe in recipes:
            st.subheader(recipe['title'])
            st.image(recipe['image'], width=200)
            st.write(f"Used Ingredients: {', '.join([ing['name'] for ing in recipe['usedIngredients']])}")
            st.write(f"Missing Ingredients: {', '.join([ing['name'] for ing in recipe['missedIngredients']])}")


from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Hugging Face model and tokenizer
def load_huggingface_model():
    model_name = "gpt2"  # You can try larger models like 'EleutherAI/gpt-neo-1.3B' for better results
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


import requests

# Generate recipe using Hugging Face API
def generate_recipe_hf_api(ingredients, dietary_pref):
    api_url = "https://api-inference.huggingface.co/models/gpt2"
    headers = {"Authorization": f"Bearer {hf_api_key}"}


    prompt = f"""
    Create a unique recipe using the following ingredients: {ingredients}.
    Make it suitable for a {dietary_pref.lower()} diet if specified.
    Include a step-by-step cooking guide.
    """

    response = requests.post(api_url, headers=headers, json={"inputs": prompt})
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        st.error("Failed to generate recipe. Please try again.")
        return None

    
if st.sidebar.button("Generate Recipe", key="generate_recipe_button"):
    st.write("Recipe generation triggered!")
    if ingredients:
        # Fetch recipes (RAG)
        recipes = fetch_recipes(ingredients, dietary_pref)
        st.subheader("Recommended Recipes:")
        for recipe in recipes:
            st.write(recipe['title'])
            st.image(recipe['image'], width=150)

        # Generate custom recipe (Hugging Face)
        custom_recipe = generate_recipe_hf_api(ingredients, dietary_pref)
        st.subheader("Your Custom Recipe:")
        st.write(custom_recipe)
