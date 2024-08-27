import os
import numpy as np
import cv2
import tensorflow as tf
import warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
data_dir = 'C:\\Personal Projects\\Fruit-Classifier\\Resources\\FruQ-multi'
model_save_path = 'C:\\Personal Projects\\Fruit-Classifier\\Resources\\Models\\fruit_classifier_model.h5'
class_save_path = 'C:\\Personal Projects\\Fruit-Classifier\\Resources\\Models\\class_indices.npy'
test_dir = 'C:\\Personal Projects\\Fruit-Classifier\\Resources\\TestImages'

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

# Load data
def load_data(data_dir):
    # Initialize an empty list to hold the data arrays
    data = []

    # List all fruit directories
    fruit_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for fruit in fruit_dirs:
        # List all condition directories for each fruit
        conditions = ['Good', 'Mild', 'Rotten']
        for condition in conditions:
            # Construct the path for each condition
            path = os.path.join(data_dir, fruit, condition)
            if os.path.isdir(path):
                # Create an array for this class
                images = []
                for img_name in os.listdir(path):
                    img_path = os.path.join(path, img_name)
                    if os.path.isfile(img_path):
                        # Read and preprocess the image
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
                        img = cv2.resize(img, (150, 150))
                        img = img.astype('float32') / 255.0
                        images.append(img)
                # Convert list to numpy array
                class_array = np.array(images)
                class_name = f'{fruit}_{condition}'
                data.append((class_name, class_array))
                print(f'Class: {class_name}, Number of Images: {len(images)}')
    
    return data

# Prepare image data generators
def prepare_data_generators(data, batch_size=32):
    # Unpack data into images and labels
    images = []
    labels = []
    
    for class_name, class_array in data:
        images.extend(class_array)
        labels.extend([class_name] * class_array.shape[0])
    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Encode labels
    label_encoder = tf.keras.layers.StringLookup()
    label_encoder.adapt(labels)
    encoded_labels = label_encoder(labels)
    
    # Convert encoded labels to one-hot encoding
    num_classes = len(label_encoder.get_vocabulary()) - 1
    one_hot_labels = tf.keras.utils.to_categorical(encoded_labels, num_classes=num_classes)
    
    # Split data into training and validation sets
    split_index = int(0.8 * len(images))
    train_images, val_images = images[:split_index], images[split_index:]
    train_labels, val_labels = one_hot_labels[:split_index], one_hot_labels[split_index:]
    
    # Create ImageDataGenerators
    datagen = ImageDataGenerator(rescale=1./255)
    
    # Ensure the data has 4D shape: (num_samples, height, width, channels)
    train_generator = datagen.flow(
        x=train_images,
        y=train_labels,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = datagen.flow(
        x=val_images,
        y=val_labels,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Create class indices dictionary from label encoder vocabulary
    class_indices = {name: index for index, name in enumerate(label_encoder.get_vocabulary())}
    
    print(f"Number of classes: {num_classes}")
    
    return train_generator, val_generator, class_indices

# Define and compile model
def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')  
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train model
def train_model(model, train_generator, val_generator, epochs=10):
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )
    return history

# Save model and class indices
def save_model_and_classes(model, class_indices, model_save_path, class_save_path):
    model.save(model_save_path)
    np.save(class_save_path, class_indices)

# Evaluate model
def evaluate_model(model, val_generator):
    # Evaluate the model
    val_steps = len(val_generator)  # Number of batches
    val_loss, val_accuracy = model.evaluate(val_generator, steps=val_steps)
    print(f'Validation Loss: {val_loss}')
    print(f'Validation Accuracy: {val_accuracy}')
    
    # Collect true labels and predictions
    y_true = []
    y_pred_prob = []

    val_generator.reset()  # Reset the generator to start from the beginning
    for batch in val_generator:
        images, labels = batch
        predictions = model.predict(images, batch_size=val_generator.batch_size)
        y_true.extend(np.argmax(labels, axis=1))  # Convert one-hot to integer labels
        y_pred_prob.extend(predictions)
        
        # Stop iteration when all data has been processed
        if len(y_true) >= val_steps * val_generator.batch_size:
            break
    
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    
    # Convert labels to binary format for ROC curve
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    
    # Compute ROC curve and AUC for each class
    plt.figure()
    for i in range(y_true_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Each Class')
    plt.legend(loc='lower right')
    plt.show()

# Test model
def test_model(model, test_dir, class_indices):
    # Ensure the test directory exists
    if not os.path.isdir(test_dir):
        raise NotADirectoryError(f"The directory {test_dir} does not exist.")
    
    # List all files in the test directory
    test_images = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
    
    if not test_images:
        print("No images found in the test directory.")
        return
    
    # Load the class labels
    class_labels = list(class_indices.keys())  # Load class labels from class indices
    
    for img_name in test_images:
        img_path = os.path.join(test_dir, img_name)
        
        # Check if the path is valid
        if not os.path.isfile(img_path):
            print(f"File not found: {img_path}")
            continue
        
        img = image.load_img(img_path, target_size=(150, 150), color_mode='rgb')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_labels[predicted_class]
        
        print(f'Image: {img_name}, Predicted Class: {predicted_label}')

# Main execution
if __name__ == "__main__":
    data = load_data(data_dir)
    train_gen, val_gen, class_indices = prepare_data_generators(data)
    num_classes = len(class_indices) - 1 # Use the length of class_indices

    model = create_model(num_classes)
    history = train_model(model, train_gen, val_gen, epochs=20)
    save_model_and_classes(model, class_indices, model_save_path, class_save_path)

    evaluate_model(model, val_gen)
    class_indices = np.load(class_save_path, allow_pickle=True).item()

    test_model(model, test_dir, class_indices)