import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import pickle
import io

from helper.functions import (
    preprocess_image
)
from helper.scrap import scrape_nutrition_data

# Load the model
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

fruits_list = ['Apel', 'Pisang', 'Alpukat', 'Ceri', 'Kiwi', 'Mangga', 'Jeruk', 'Nanas', 'Stroberi', 'Semangka']

def prepare_image_from_bytes(image_bytes):
    """
    Process image directly from bytes and predict fruit/vegetable class
    """
    try:
        # Preprocess the image for prediction
        image_array = preprocess_image(image_bytes)
        
        # Make prediction using the model
        prediction = model.predict(image_array)
        
        # Get the predicted class index
        pred_idx = np.argmax(prediction[0])
        
        # Get the confidence score
        # confidence = float(prediction[0][pred_idx])
        
        # Get the food name
        fruit_name = fruits_list[pred_idx] if pred_idx < len(fruits_list) else "unknown"
        
        return fruit_name
    except Exception as e:
        st.error(f"Error predicting image: {str(e)}")
        return None

def run():
    # Set up the UI
    st.title("Fruits🍍 Classification")
    st.write("Upload an image of a fruit or vegetable to classify it")
    
    # Upload image through streamlit interface
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])
    
    if img_file is not None:
        # Display the uploaded image
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_container_width=False, caption="Uploaded Image")
        
        # Add a prediction button
        if st.button("Predict"):
            # Show a spinner while processing
            with st.spinner("Analyzing image..."):
                # Get image bytes directly from uploaded file
                img_file.seek(0)  # Reset file pointer to beginning
                image_bytes = img_file.read()
                
                # Process image directly from bytes
                result = prepare_image_from_bytes(image_bytes)
                
                if result:
                    # Display prediction result
                    st.success(f"**Predicted : {result}**")
                    
                    # Display calorie information if available
                    cal = scrape_nutrition_data(result)
                    if cal:
                        st.warning(f"**{cal} (100 grams)**")
                    else:
                        st.warning("**Calorie information not available**")


# Run the application
if __name__ == "__main__":
    run()
