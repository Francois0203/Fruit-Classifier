import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

SCRIPT_PATH = os.getcwd()
MODEL_DIR = os.path.join(SCRIPT_PATH, "Resources", "Models")
RESULTS_DIR = os.path.join(SCRIPT_PATH, "Resources", "Results")
DATASET_DIR = os.path.join(SCRIPT_PATH, "Resources", "FruQ-multi Class Split")

print("\n")
print("Script path: ", SCRIPT_PATH)
print("Data set path: ", DATASET_DIR)
print("Model path: ", MODEL_DIR)
print("Results saved at: ", RESULTS_DIR)
print("\n")

def load_and_preprocess_data(dataset_dir):
    #Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,  # 20% of data for validation
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        channel_shift_range=0.2  # Simulates color noise
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Load training data 
    train_generator = train_datagen.flow_from_directory(
        directory=dataset_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Load validation data
    validation_generator = validation_datagen.flow_from_directory(
        directory=dataset_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Save the class labels to a JSON file for later use
    with open(os.path.join(MODEL_DIR, "class_labels.json"), 'w') as json_file:
        json.dump(train_generator.class_indices, json_file)
    
    return train_generator, validation_generator

#Moedl Architecture
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Slower learning rate
                loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#Training
def train_model(model, train_generator, validation_generator):
    checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.keras"), monitor='val_accuracy', save_best_only=True, mode='max')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # Plot accuracy vs epochs
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy vs Epochs')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_vs_epochs.png"))

    return history

#Evaluation
def evaluate_model(model, validation_generator):
    validation_generator.reset()
    Y_pred = model.predict(validation_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = validation_generator.classes
    
    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Overall Precision: {precision:.4f}")
    print(f"Overall Recall: {recall:.4f}")
    print(f"Overall F1-Score: {f1:.4f}")

    y_true_binary = label_binarize(y_true, classes=np.arange(len(validation_generator.class_indices)))
    fpr, tpr, _ = roc_curve(y_true_binary.ravel(), Y_pred.ravel())
    roc_auc = auc(fpr, tpr)
    print(f"Overall AUC: {roc_auc:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Overall ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))

train_generator, validation_generator = load_and_preprocess_data(DATASET_DIR)
    
input_shape = (150, 150, 3)
num_classes = 33  
model = build_model(input_shape, num_classes)
    
history = train_model(model, train_generator, validation_generator)
    
evaluate_model(model, validation_generator)