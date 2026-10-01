import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# CONFIGURATION
# =========================

MODEL_PATH = "models/vehicle_mobilenetv2_final.keras"

CLASS_NAMES = [
    "SUV",
    "bus",
    "convertible",
    "coupes",
    "hatchback",
    "pickup",
    "sedan",
    "station_wagon",
    "trucks",
    "van"
]


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# =========================
# PAGE
# =========================

st.set_page_config(
    page_title="Vehicle Type Classifier",
    page_icon="🚗",
    layout="centered"
)

st.title("Vehicle Type Classifier")
st.write("Upload an image of a vehicle to classify its body type.")


# =========================
# UPLOAD IMAGE
# =========================

uploaded_file = st.file_uploader(
    "Upload vehicle image",
    type=["jpg", "jpeg", "png"]
)


# =========================
# PREDICTION
# =========================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict Vehicle"):

        # Resize image
        resized_image = image.resize((224, 224))

        # Convert to array
        image_array = np.array(resized_image)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Prediction
        predictions = model.predict(
            image_array,
            verbose=0
        )

        predicted_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_index] * 100

        predicted_class = CLASS_NAMES[predicted_index]

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )