import tensorflow as tf
import tensorflow as tf
from PIL import Image
import numpy as np



MODEL_PATH = "models/vehicle_mobilenetv2_final.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("Model berhasil dimuat.")



MODEL_PATH = "models/vehicle_mobilenetv2_final.keras"

class_names = [
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

model = tf.keras.models.load_model(MODEL_PATH)

print("Model berhasil dimuat.")

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index] * 100

    predicted_class = class_names[predicted_index]

    print(f"Prediction : {predicted_class}")
    print(f"Confidence : {confidence:.2f}%")
    

image_path = "test.jpg"

predict_image(image_path)