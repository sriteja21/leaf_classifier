import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Load the model
from tensorflow.keras.applications.efficientnet import preprocess_input
model = tf.keras.models.load_model('cnn_best')

# Constants
LABELS = ['Apple', 'Blueberry', 'Cherry', 'Corn', 'Grape', 'Orange', 'Peach', 'Pepper', 'Potato', 'Raspberry', 'Soybean', 'Squash', 'Strawberry', 'Tomato']
INPUT_SHAPE = (256, 256, 3)
IMAGE_SIZE = (INPUT_SHAPE[0], INPUT_SHAPE[1])
NUM_CLASSES = len(LABELS)

def preprocess_image(img):
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)   # Normalization
    return img_array

def classify_leaf(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = LABELS[predicted_class_index]
    return predicted_class

# Streamlit application
st.title("Leaf Classifier")

# Setting CSS style to align text in the middle
st.markdown(
    """
    <style>
    /* Align text in the middle */
    .stApp, .stMarkdown, .stTextInput > div, .stTextInput > div > div {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

uploaded_img = st.empty()

# Upload image and classify
uploaded_file = st.file_uploader("Choose a leaf image...", type="jpg")
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    uploaded_img.image(img, caption='Uploaded Image.', use_column_width=True)
    if st.button("Classify"):
        predicted_class = classify_leaf(img)
        st.write(f"Predicted class: {predicted_class}")

# Clear button
if st.button("Clear"):
    uploaded_img.empty()
