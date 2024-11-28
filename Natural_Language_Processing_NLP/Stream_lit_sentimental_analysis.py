import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the pre-trained model and tokenizer
MODEL_PATH = "Natural_Language_Processing_NLP/bert-finetuned/"
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
