# Write a python script that reads images from files. The main file is called 'FruQ-multi' saved at 'C:\Personal Projects\Fruit-Classifier\Resources\FruQ-multi'. This folder has sub folders with fruit names. Then each fruit has 3 folders in it: Good, Mild, Rotten. These images should be extracted and saved into arrays which will be the different classes, for example: banana_mild, apple_good, etc. The arrays should then be used to train a CNN model, and save that model with its labels in a json file at the following location: 'C:\Personal Projects\Fruit-Classifier\Resources\Models'. The training process should choose the optimal hyperparameters to get the best performance. The model should flip the images during training as well.

# The following info should be displayed during the whole process: F1 score, AUC score, ROC plot, confusion matrix, accuracy, etc. The info should be easy to read and the graphs should be beautiful. 

# The file structure is as follows:

# FruQ-multi
# -----fruit name
# ----------Good
# ---------------images.png
# ----------Mild
# ---------------images.png
# ----------Rotten
# ---------------images.png

# There should be 33 classes. The test images are stored at: 'C:\Personal Projects\Fruit-Classifier\Resources\TestImages'. 

# Each task should be written in a separate algorithm so that i can easily edit it and modify paramteres.






import os
import json
import numpy as np
import tensorflow as tf
import warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = 'C:\\Personal Projects\\Fruit-Classifier\\Resources\\FruQ-multi Class Split'
TEST_DIR = 'C:\\Personal Projects\\Fruit-Classifier\\Resources\\TestImages'
MODEL_SAVE_PATH = 'C:\\Personal Projects\\Fruit-Classifier\\Resources\\Models\\fruit_classifier.keras'
LABELS_SAVE_PATH = 'C:\\Personal Projects\\Fruit-Classifier\\Resources\\Models\\fruit_labels.json'

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')


# Step 1: Load and preprocess the images
def load_data(base_dir):
    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

# Step 2: Build the CNN model
def build_model(input_shape, num_classes):
    model = Sequential([
        # First Convolutional Block
        Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Third Convolutional Block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Fourth Convolutional Block
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Fifth Convolutional Block
        Conv2D(512, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Flatten and Fully Connected Layers
        Flatten(),
        
        Dense(1024, activation='relu'),           
        Dropout(0.5),
        BatchNormalization(),
        
        Dense(512, activation='relu'),           
        Dropout(0.5),
        BatchNormalization(),
        
        Dense(256, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        
        # Output Layer
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 3: Train the model
def train_model(model, train_generator, validation_generator):
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=50,
        callbacks=[early_stop, checkpoint]
    )
    return history

# Step 4: Evaluate the model
def evaluate_model(model, validation_generator):
    # Get the true labels
    Y_val = validation_generator.classes  # True labels
    
    # Predict the probabilities for each class
    Y_pred = model.predict(validation_generator)
    
    # One-hot encode the true labels
    num_classes = len(validation_generator.class_indices)
    Y_val_one_hot = label_binarize(Y_val, classes=list(range(num_classes)))
    
    # Print classification report
    print("Classification Report:")
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    print(classification_report(Y_val, Y_pred_classes, labels=list(range(num_classes)), target_names=list(validation_generator.class_indices.keys())))
    
    # Print confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(Y_val, Y_pred_classes, labels=list(range(num_classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(), yticklabels=validation_generator.class_indices.keys())
    plt.show()
    
    # Calculate ROC AUC score for each class
    print("ROC AUC Score:")
    auc_scores = []
    for i in range(Y_pred.shape[1]):
        fpr, tpr, _ = roc_curve(Y_val_one_hot[:, i], Y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC: {roc_auc:.2f})')

    # Plotting the ROC curves
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    # Calculate the macro average AUC
    auc_macro = np.mean(auc_scores)
    print(f"Macro AUC: {auc_macro:.4f}")
    
    # Calculate accuracy
    print("Accuracy Score:")
    accuracy = accuracy_score(Y_val, Y_pred_classes)
    print(f"Accuracy: {accuracy:.4f}")

    return auc_macro, accuracy

# Step 5: Save the model and labels
def save_model_and_labels(model, train_generator):
    model.save(MODEL_SAVE_PATH)
    labels = {v: k for k, v in train_generator.class_indices.items()}
    with open(LABELS_SAVE_PATH, 'w') as f:
        json.dump(labels, f)
    print(f"Model and labels saved.")

# Step 6: Main function to run all steps
def __main__():
    train_generator, validation_generator = load_data(BASE_DIR)
    model = build_model(input_shape=(128, 128, 3), num_classes=len(train_generator.class_indices))
    history = train_model(model, train_generator, validation_generator)
    auc, accuracy = evaluate_model(model, validation_generator)
    save_model_and_labels(model, train_generator)

if __name__ == "__main__":
    __main__()
