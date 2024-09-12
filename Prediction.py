import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
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
        master.geometry("500x600")
        master.config(bg="#f0f0f0")  # Set background color

        # Frame for image and buttons
        self.frame = Frame(master, bg="#ffffff", bd=2, relief=tk.RIDGE, padx=10, pady=10)
        self.frame.pack(pady=20)

        # Add label
        self.label = Label(self.frame, text="Upload an image to classify", font=("Helvetica", 16), bg="#ffffff")
        self.label.pack(pady=10)

        # Upload button
        self.upload_button = Button(self.frame, text="Upload Image", command=self.upload_image, font=("Helvetica", 14),
                                    bg="#4CAF50", fg="white", bd=3, relief=tk.RAISED)
        self.upload_button.pack(pady=10)

        # Predict button (disabled initially)
        self.predict_button = Button(self.frame, text="Predict", command=self.predict, state=tk.DISABLED, 
                                    font=("Helvetica", 14), bg="#008CBA", fg="white", bd=3, relief=tk.RAISED)
        self.predict_button.pack(pady=10)

        # Display the uploaded image
        self.image_label = Label(self.frame, bg="#ffffff", relief=tk.SUNKEN)
        self.image_label.pack(pady=10)

        # Result label to display predictions
        self.result_label = Label(master, text="", font=("Helvetica", 14), bg="#f0f0f0", fg="#333333")
        self.result_label.pack(pady=20)

        # Initialize image path
        self.image_path = None

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            img = Image.open(file_path)
            img = img.resize((250, 250))  # Resize image for display
            img = ImageTk.PhotoImage(img)

            self.image_label.config(image=img, width=250, height=250)
            self.image_label.image = img  # Store image to prevent garbage collection

            self.predict_button.config(state=tk.NORMAL)  # Enable predict button after image upload

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
            self.result_label.config(
                text=f"Prediction: {predicted_label}\nConfidence: {confidence * 100:.2f}%",
                fg="#333333"
            )

# Create main window
root = tk.Tk()
app = PredictionApp(root)
root.mainloop()
