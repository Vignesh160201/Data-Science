import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# Load the saved model
MODEL_PATH = 'mobilenetv2_beans_model.keras'
model = load_model(MODEL_PATH)

# Define the class names (replace with your actual class names)
class_names = ['Healthy', 'Angular Leaf Spot', 'Bean Rust']

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)  # Resize to model's expected input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to load an image from a URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for invalid responses
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Streamlit App
st.title("Bean Disease Classification")
st.write("Upload an image or provide an image link to classify the condition of a bean leaf.")

# Sidebar for input options
option = st.sidebar.selectbox("Choose an input method:", ["Upload Image", "Image URL"])

if option == "Upload Image":
    # File uploader
    uploaded_file = st.file_uploader("Choose a bean leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

elif option == "Image URL":
    # Input box for image URL
    image_url = st.text_input("Enter the image URL:")
    if image_url:
        image = load_image_from_url(image_url)
        if image:
            st.image(image, caption="Image from URL", use_column_width=True)

# Prediction
if 'image' in locals() and image:
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make a prediction
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display the result
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
else:
    st.write("Please upload an image or enter a valid image URL.")
