import os
import json
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# Set the path to the dataset
DATASET_PATH = r'C:\Personal Projects\Fruit-Classifier\Resources\FQ'
MODEL_PATH = r'C:\Personal Projects\Fruit-Classifier\Resources\Models'

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

# Function to load and preprocess the data
def load_and_preprocess_data(img_size=(150, 150), batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.2,
        horizontal_flip=True,  
        vertical_flip=True     
    )
    
    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False  # Do not shuffle for evaluation
    )
    
    return train_generator, validation_generator

# Function to build the neural network model
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
    
    return model

# Function to compile the model
def compile_model(model, learning_rate=0.001):
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
    
    return model

# Function to train the model
def train_model(model, train_generator, validation_generator, epochs=10):
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        validation_data=validation_generator,
        epochs=epochs
    )
    
    return history

# Function to evaluate the model
def evaluate_model(model, validation_generator):
    loss, accuracy = model.evaluate(validation_generator)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    return loss, accuracy

def save_model(model, name):
    name = name + '.h5'
    model_path = os.path.join(MODEL_PATH, name)
    model.save(model_path)
    print(f"Model saved to {MODEL_PATH}")

# Function to save class labels
def save_class_labels(class_indices):
    labels = {v: k for k, v in class_indices.items()}
    model_path = os.path.join(MODEL_PATH, 'class_labels.json')

    with open(model_path, 'w') as f:
        json.dump(labels, f)
    print(f"Class labels saved to {model_path}")

# Function to compute additional metrics and plot ROC curves
def compute_metrics_and_plot_roc(model, validation_generator):
    y_true = validation_generator.classes
    y_pred = model.predict(validation_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=len(validation_generator.class_indices))

    # Compute classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=list(validation_generator.class_indices.keys())))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Compute ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(len(validation_generator.class_indices)):
        fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(validation_generator.class_indices))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(validation_generator.class_indices)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(validation_generator.class_indices)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot ROC curves
    plt.figure(figsize=(12, 8))
    for i in range(len(validation_generator.class_indices)):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
    plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average ROC curve (area = {roc_auc["macro"]:0.2f})', linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Compute overall AUC score
    auc_value = roc_auc_score(y_true_one_hot, y_pred, average="macro")
    print(f'\nOverall AUC Score: {auc_value:.2f}')
    
    return auc_value

def __main__():
    img_size = (150, 150)
    batch_size = 32
    epochs = 10

    # Load and preprocess data
    train_generator, validation_generator = load_and_preprocess_data(img_size, batch_size)
    
    # Build the model
    input_shape = (*img_size, 3)
    num_classes = len(train_generator.class_indices)
    model = build_model(input_shape, num_classes)
    
    # Compile the model
    model = compile_model(model)
    
    # Train the model
    history = train_model(model, train_generator, validation_generator, epochs)
    
    # Evaluate the model
    evaluate_model(model, validation_generator)
    
    # Compute metrics and plot ROC curves
    compute_metrics_and_plot_roc(model, validation_generator)

    # Save model and class labels
    save_model(model, 'first')
    save_class_labels(train_generator.class_indices)

# Main script to run the functions
if __name__ == "__main__":
    __main__()