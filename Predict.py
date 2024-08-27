import os
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = r'C:\Personal Projects\Fruit-Classifier\Resources\Models'
TEST_IMAGES = r'C:\Personal Projects\Fruit-Classifier\Resources\TestImages'

# Load the trained model
def load_trained_model(name='first'):
    name = name + '.h5'
    model_path = os.path.join(MODEL_PATH, name)
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model

# Function to load class labels
def load_class_labels(model_path=MODEL_PATH):
    filename = os.path.join(model_path, 'class_labels.json')
    with open(filename, 'r') as f:
        labels = json.load(f)
    return labels

# Predict the class of a new image
def predict_image(model, img_path, img_size=(150, 150)):
    img = image.load_img(img_path, target_size=img_size, color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions, axis=1)[0]
    class_labels = load_class_labels()

    # Convert the integer index to a string
    class_label = class_labels[str(class_idx)]
    print(f"Predicted class: {class_label}")
    return class_label

def __main__():
    model = load_trained_model(name='first')
    class_labels = load_class_labels() 
    print(class_labels)
    img_path = os.path.join(TEST_IMAGES, 'strawberry_1.png')  
    predict_image(model, img_path)

if __name__ == "__main__":
    __main__()