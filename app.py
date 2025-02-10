
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from tensorflow import keras
from keras.models import load_model 
from keras.preprocessing.image import img_to_array 
import os


model = load_model(r'D:\yamuna\Skill4_CNN\CNN_waste _classification.keras')  

def preprocess_image(image):
    img_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array / 255.0  
    img_array = img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("Waste Classification App")
st.write("Upload an image of waste to classify it.")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    class_labels = ['Class 0', 'Class 1'] 
    predicted_class = class_labels[np.argmax(predictions)]


    st.write(f"Predicted Class: {predicted_class}")