import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
import gdown

file_id_from_drive ="1SPogm0YWHWmVE-Or0ZJ7fDTcZ_8TzB65"
# Google Drive file IDs for model files
DRIVE_FILES = {
    "pytorch_model.bin": file_id_from_drive,
    "config.json": file_id_from_drive,
    "tokenizer.json": file_id_from_drive
}

MODEL_PATH = "./bert-finetuned"

# Function to download files from Google Drive
def download_model_files():
    os.makedirs(MODEL_PATH, exist_ok=True)
    for file_name, file_id in DRIVE_FILES.items():
        file_path = os.path.join(MODEL_PATH, file_name)
        if not os.path.exists(file_path):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, file_path, quiet=False)
            print(file_path)

# Load the pre-trained model and tokenizer
download_model_files()
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Function to predict sentiment
def predict_sentiment(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted class
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Map class to sentiment
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[predicted_class]

# Streamlit App Layout
st.title("Sentiment Analysis App")
st.write("Enter a product review to predict its sentiment.")

# Input from the user
user_input = st.text_area("Write your review here:")

# Predict sentiment when the user clicks the button
if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.subheader(f"Predicted Sentiment: {sentiment}")
    else:
        st.error("Please enter a review to analyze.")

# Footer
st.write("---")
st.write("Built with [Streamlit](https://streamlit.io/) and Transformers.")
