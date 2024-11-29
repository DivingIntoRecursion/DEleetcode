from huggingface_hub import login

# Add your Hugging Face token here
huggingface_api_token = "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"

# Login to Hugging Face with the token
login(token=huggingface_api_token)

import os

# Set the Hugging Face token directly in the environment
os.environ["HF_TOKEN"] = huggingface_api_token

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd

# Load the Hugging Face model and tokenizer (use a lightweight model like distilgpt2 or gpt2)
model_name = "distilgpt2"  # You can change this to "gpt2", "EleutherAI/gpt-neo-1.3B", etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate a response
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")  # Convert input to tensor
    outputs = model.generate(
        inputs, 
        max_length=150, 
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_k=50,
        max_new_tokens=100  # Ensure we're generating new content
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit Web Interface
st.title("AI Food Distribution Assistant")
st.write("Welcome to the AI-powered Food Distribution Assistant.")

# Sidebar Navigation for Different Features
menu = ["Automate Communication", "Training Guide", "Community Insights", "Chatbot"]
choice = st.sidebar.selectbox("Select a Feature", menu)

# Feature 1: Automating Communication
if choice == "Automate Communication":
    st.subheader("Automate Communication")
    user_type = st.selectbox("Select User Type", ["Donor", "Volunteer", "Recipient"])
    name = st.text_input("Name")
    food_quantity = st.text_input("Food Quantity (e.g., 100 meals)")
    delivery_time = st.text_input("Delivery Time (e.g., 3 PM)")
    volunteer_name = st.text_input("Volunteer Name")
    phone_number = st.text_input("Phone Number")
    language = st.selectbox("Language", ["English (en)", "Hindi (hi)", "Tamil (ta)", "Bengali (bn)"])

    if st.button("Generate Message"):
        prompt = f"""
        Generate a personalized {user_type} message in {language}.
        Name: {name}, Food Quantity: {food_quantity}, Delivery Time: {delivery_time},
        Volunteer: {volunteer_name}, Contact: {phone_number}.
        """
        response = generate_response(prompt)
        st.write("Generated Message:", response)

# Feature 2: Training Volunteers
elif choice == "Training Guide":
    st.subheader("Training Volunteers")
    region = st.text_input("Region (e.g., Mumbai)")
    food_type = st.selectbox("Food Type", ["Perishable", "Non-Perishable"])

    if st.button("Generate Training Guide"):
        prompt = f"""
        Create a step-by-step training guide for volunteers handling {food_type} food in {region}.
        Include checking food quality, packaging, and delivery instructions.
        """
        response = generate_response(prompt)
        st.write("Training Guide:", response)

# Feature 3: Generating Community Insights
elif choice == "Community Insights":
    st.subheader("Community Insights")

    # Upload sample CSV or generate sample data
    sample_data = {
        'Region': ['Sector 5', 'Sector 7', 'Sector 12'],
        'Population': [500, 300, 450],
        'Malnutrition Rate': [0.25, 0.35, 0.20],
        'Vegetarian': [0.7, 0.8, 0.6]
    }
    df = pd.DataFrame(sample_data)
    st.write("Sample Data:", df)

    if st.button("Generate Insights"):
        prompt = f"""
        Analyze the following data and generate an action plan for food distribution:
        {df.to_string(index=False)}
        Focus on priority areas, dietary preferences, and challenges.
        """
        response = generate_response(prompt)
        st.write("Community Insights:", response)

# Feature 4: Chatbot
elif choice == "Chatbot":
    st.subheader("Interactive Chatbot")
    user_query = st.text_input("Ask a question")

    if st.button("Get Response"):
        # Updated system message to improve model behavior
        system_message = "You are a helpful assistant for food distribution, answering questions related to delivery locations, schedules, and other logistics. Please provide clear, relevant answers based on the user's query."
        
        # Combine system message with user query
        prompt = system_message + " " + user_query
        response = generate_response(prompt)
        st.write("AI Response:", response)