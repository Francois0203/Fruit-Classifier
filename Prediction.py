import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os
import json

# Load the model and class labels
MODEL_PATH = os.path.join(os.getcwd(), "Resources", "Models", "best_model.keras")
LABELS_PATH = os.path.join(os.getcwd(), "Resources", "Models", "class_labels.json")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load labels
with open(LABELS_PATH, 'r') as json_file:
    class_labels = json.load(json_file)
    class_labels = {v: k for k, v in class_labels.items()}  # Reverse mapping

# Create the GUI application
class PredictionApp:
    def __init__(self, master):
        self.master = master
        master.title("Fruit Classifier")

        self.label = Label(master, text="Upload an image to classify")
        self.label.pack()

        self.upload_button = Button(master, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

        self.predict_button = Button(master, text="Predict", command=self.predict, state=tk.DISABLED)
        self.predict_button.pack()

        self.image_label = Label(master)
        self.image_label.pack()

        self.result_label = Label(master, text="")
        self.result_label.pack()

        self.image_path = None

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            img = Image.open(file_path)
            img = img.resize((150, 150))  # Resize to model input size
            img = ImageTk.PhotoImage(img)

            self.image_label.config(image=img)
            self.image_label.image = img

            self.predict_button.config(state=tk.NORMAL)

    def predict(self):
        if self.image_path:
            img = tf.keras.preprocessing.image.load_img(self.image_path, target_size=(150, 150))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize image

            # Predict using the model
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions)

            # Retrieve class label from the saved labels
            predicted_label = class_labels.get(predicted_class, "Unknown")

            # Update the result label
            self.result_label.config(text=f"Prediction: {predicted_label} ({confidence*100:.2f}% confidence)")

# Create main window
root = tk.Tk()
app = PredictionApp(root)
root.mainloop()
