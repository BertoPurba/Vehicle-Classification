# Vehicle Type Classification

A deep learning project for classifying vehicle body types using **MobileNetV2** and **TensorFlow/Keras**. The project also includes a **Streamlit web application** for testing vehicle images.

## Project Overview

This project uses transfer learning with MobileNetV2 to classify vehicle images into 10 different vehicle types.

The model receives an image of a vehicle and predicts its vehicle body type along with the prediction confidence.

## Vehicle Classes

The model can classify 10 vehicle types:

* SUV
* Bus
* Convertible
* Coupes
* Hatchback
* Pickup
* Sedan
* Station Wagon
* Trucks
* Van

## Dataset

The dataset used in this project is:

**Vehicle Type 10: A Dataset for Vehicle Body Type Classification**

The dataset contains images of different vehicle body types and is used for training and validation.

## Model

The project uses **MobileNetV2** with transfer learning.

### Architecture

```text
Input Image
     ↓
224 × 224 × 3
     ↓
MobileNetV2
     ↓
Global Average Pooling
     ↓
Dropout
     ↓
Dense Layer (128)
     ↓
Dropout
     ↓
Dense Layer (10)
     ↓
Vehicle Type
```

The pretrained MobileNetV2 feature extractor was used as the base model, followed by custom classification layers.

## Data Augmentation

Data augmentation was applied during training to improve model generalization.

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])
```

## Model Performance

The final model achieved:

| Metric              |     Result |
| ------------------- | ---------: |
| Validation Accuracy | **86.79%** |
| Validation Loss     | **0.3726** |

The classification report showed an overall accuracy of approximately **87%** on the validation set.

## Example Prediction

The model was also tested using a vehicle image outside the validation examples.

Example:

```text
Prediction : trucks
Confidence : 99.99%
```

## Web Application

A Streamlit application was created to make the model easier to test.

The application allows users to:

1. Upload a vehicle image
2. Preview the uploaded image
3. Run the classification model
4. View the predicted vehicle type
5. View the prediction confidence

### Run the application

First, activate the virtual environment and install the required dependencies.

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app.py
```

The application will open in the browser.

## Project Structure

```text
Klasifikasi Kendaraan/
│
├── models/
│   └── vehicle_mobilenetv2_final.keras
│
├── dataset/
│
├── app.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Technologies

* Python
* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Pillow
* Streamlit
* Matplotlib
* Scikit-learn

## Future Improvements

Possible improvements for this project include:

* Increasing the amount of training data
* Improving classification of visually similar vehicle types
* Further hyperparameter tuning
* Testing other pretrained CNN architectures
* Improving the Streamlit interface
* Deploying the application online

## Author

**Berto Jdoyvan Purba**

GitHub: [BertoPurba](https://github.com/BertoPurba)
