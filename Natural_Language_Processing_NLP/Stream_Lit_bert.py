import streamlit as st
import gdown
import os
import zipfile
from transformers import BertTokenizer, BertForSequenceClassification
import torch
# Streamlit App Layout

# Title
st.title("Sentiment Analysis App")
st.write("Provide a Google Drive link for Bert model.")

# Input for Google Drive shareable link
drive_link = st.text_input("Enter Google Drive Shareable Link to Zipped Folder:")

st.write("Enter a product review to predict its sentiment.")
# Text input for predictions
user_text = st.text_area("Enter text for classification:")


# Button to process
if st.button("Load Folder and Predict"):
    if drive_link:
        try:
            # Extract file ID
            file_id = drive_link.split('/d/')[1].split('/')[0]
            file_url = f"https://drive.google.com/uc?id={file_id}"

            # Download the ZIP file
            zip_path = "bert-finetuned.zip"
            st.write("Downloading the zipped folder...")
            gdown.download(file_url, zip_path, quiet=False)

            # Extract ZIP file
            st.write("Extracting files...")
            extract_dir = "bert-finetuned"
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            st.success("Folder downloaded and extracted successfully!")
            st.write(f"Files in model directory: {os.listdir(extract_dir)}")

            # Load BERT model and tokenizer
            st.write("Loading BERT tokenizer and model...")
            tokenizer = BertTokenizer.from_pretrained(extract_dir)
            model = BertForSequenceClassification.from_pretrained(extract_dir)
            st.success("Model and tokenizer loaded successfully!")

            # Perform prediction
            if user_text.strip():
                st.write("Performing text classification...")
                inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    prediction = torch.argmax(outputs.logits, dim=1).item()

                #st.success(f"Predicted Class: {prediction}")
                sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
                st.subheader(f"Predicted Sentiment: {sentiment_map[prediction]}")
                
                
            else:
                st.error("Please enter a review to analyze.")

            # Cleanup
            os.remove(zip_path)

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a valid Google Drive shareable link.")

# Footer
st.write("---")
st.write("Built with [Streamlit](https://streamlit.io/) and Transformers.")
