import streamlit as st
import gdown
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Title
st.title("BERT Model for Text Classification")
st.write("Load a BERT model and tokenizer from Google Drive to perform text classification.")

# Input for Google Drive link
drive_link = st.text_input("Enter Google Drive Shareable Link to the Model Folder:")

# Text input for classification
user_text = st.text_area("Enter text for classification:", "")

# Button to load model and predict
if st.button("Load Model and Predict"):
    if drive_link:
        try:
            # Extract file ID from the Google Drive link
            file_id = drive_link.split('/d/')[1].split('/')[0]
            file_url = f"https://drive.google.com/uc?id={file_id}"
            
            # Create directory to store the model files
            model_dir = "bert_model"
            os.makedirs(model_dir, exist_ok=True)

            # Download the BERT model folder as a ZIP file
            st.write("Downloading the model files, please wait...")
            output_zip = "bert_model.zip"
            gdown.download(file_url, output_zip, quiet=False)

            # Unzip the model folder
            import zipfile
            with zipfile.ZipFile(output_zip, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            st.success("Model downloaded and extracted successfully!")

            # Load the tokenizer and model
            st.write("Loading BERT tokenizer and model...")
            tokenizer = BertTokenizer.from_pretrained(model_dir)
            model = BertForSequenceClassification.from_pretrained(model_dir)
            st.success("Model and tokenizer loaded successfully!")

            # Clean up ZIP file
            os.remove(output_zip)

            # Make predictions
            if user_text.strip():
                st.write("Performing text classification...")
                inputs = tokenizer(user_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    predictions = torch.argmax(outputs.logits, dim=1).item()
                
                st.success(f"Predicted Class: {predictions}")
            else:
                st.warning("Please enter some text for classification.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid Google Drive shareable link.")
