
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("face_mask_model.h5")

model = load_model()

# Preprocess function
def preprocess_image(img):
    img = img.resize((128, 128))  # Assuming model was trained on 128x128
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

st.title("😷 Face Mask Detection")
st.write("Upload an image or take a photo to check if a person is wearing a mask.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_names = ["Mask", "No Mask"]
    predicted_class = class_names[int(prediction[0][0] > 0.5)]
    st.subheader(f"Prediction: {predicted_class}")

# Optional webcam capture (if running locally)
if st.button("Use Webcam"):
    cap = cv2.VideoCapture(0)
    st.write("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)
        processed_image = preprocess_image(img_pil)
        prediction = model.predict(processed_image)
        class_names = ["Mask", "No Mask"]
        predicted_class = class_names[int(prediction[0][0] > 0.5)]
        cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
