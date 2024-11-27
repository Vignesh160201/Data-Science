import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the saved model
#MODEL_PATH = 'C:/Users/Vignesh/Downloads/SRM/Sem/Deep Learning/mobilenetv2_beans_model.keras'
MODEL_PATH = 'mobilenetv2_beans_model.keras'
model = load_model(MODEL_PATH)

# Define the class names (replace with your actual class names)
class_names = ['Healthy', 'Angular Leaf Spot', 'Bean Rust']

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((128, 128))  # Resize to model's expected input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit App
st.title("Bean Disease Classification")
st.write("Upload an image of a bean leaf to identify its health condition.")

# File uploader
uploaded_file = st.file_uploader("Choose a bean leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make a prediction
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Display the result
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Error handling for unsupported file types
else:
    st.write("Please upload a valid image file.")
