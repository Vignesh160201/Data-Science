import streamlit as st
import gdown
import os
import zipfile
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Title
st.title("Access Entire Folder from Google Drive")
st.write("Provide a Google Drive link to a zipped folder and use BERT for predictions.")

# Input for Google Drive shareable link
drive_link = st.text_input("Enter Google Drive Shareable Link to Zipped Folder:")

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
            zip_path = "model_folder.zip"
            st.write("Downloading the zipped folder...")
            gdown.download(file_url, zip_path, quiet=False)

            # Extract ZIP file
            st.write("Extracting files...")
            extract_dir = "model_folder"
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            st.success("Folder downloaded and extracted successfully!")

            # Load BERT model and tokenizer
            st.write("Loading BERT tokenizer and model...")
            tokenizer = BertTokenizer.from_pretrained(extract_dir)
            model = BertForSequenceClassification.from_pretrained(extract_dir)
            st.success("Model and tokenizer loaded successfully!")

            # Perform prediction
            if user_text.strip():
                st.write("Performing text classification...")
                inputs = tokenizer(us
