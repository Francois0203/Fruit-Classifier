import sys, os, time, datetime # Standard system libraries
import cv2 # Image processing library
import numpy as np # Transform data with this library
from tensorflow import keras
from keras.models import load_model
import pickle # Used to retrieve model dictionaries in this case   

sys.path.append(os.getcwd()) # Set working directory

# Load pre-trained model
model = load_model('Resources/Models/test.h5')
print("Successfully loaded model")

# Access dictionaries from model
with open('Resources/Models/lookup.pkl', 'rb') as f:
    lookup = pickle.load(f)
with open('Resources/Models/reverselookup.pkl', 'rb') as f:
    reverselookup = pickle.load(f)

# Function to preprocess image
def preprocess_image(img):
    img = cv2.resize(img, (320, 120))  # Resize

    arr = np.array(img, dtype = 'float32')
    arr = arr.reshape((1, 120, 320, 1))  # Add batch dimension
    arr /= 255.0  # Normalize

    return arr

# Predict the gesture name of the image using the model
def predict_fruit_category(image_data):
    predictions = model.predict(image_data)
    predicted_indx = np.argmax(predictions[0])
    predicted_gesture = reverselookup[predicted_indx]
    score = float("%0.2f" % (max(predictions[0]) * 100))

    return predicted_gesture, score

def __main__():
    image_path = "Resources/TestImages/strawberry_1.png"
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

    image_data = preprocess_image(gray)

    # Make prediction with the model
    predicted_gesture, score = predict_fruit_category(image_data)

    # Print output
    print(f"Predicted Fruit: {predicted_gesture}, Confidence: {score}%") 

if __name__ == '__main__':
    __main__()