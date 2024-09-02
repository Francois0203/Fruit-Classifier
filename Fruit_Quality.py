import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

# Define device (use GPU if available, otherwise use CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Check if the GPU is available and being used
if device.type == 'cuda':
    print(f"GPU is available and being used: {torch.cuda.get_device_name(0)}")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
else:
    print("Using CPU")

# Define the data directory
data_dir = r'C:\Personal Projects\Fruit-Classifier\Resources\FruQ-multi'

# Custom Dataset Class to handle both fruit type and ripeness labels
class FruitRipenessDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []  # List to store paths of all images
        self.fruit_labels = []  # List to store labels for fruit types
        self.ripeness_labels = []  # List to store labels for ripeness
        self.fruit_classes = []  # List of unique fruit classes
        self.ripeness_classes = ['Fresh', 'Mild', 'Rotten']  # Predefined ripeness classes

        # Loop through directories to populate image paths and labels
        for fruit_class in os.listdir(root_dir):
            fruit_path = os.path.join(root_dir, fruit_class)
            if os.path.isdir(fruit_path):
                self.fruit_classes.append(fruit_class)
                for ripeness_class in os.listdir(fruit_path):
                    ripeness_path = os.path.join(fruit_path, ripeness_class)
                    if os.path.isdir(ripeness_path):
                        for img_name in os.listdir(ripeness_path):
                            img_path = os.path.join(ripeness_path, img_name)
                            self.image_paths.append(img_path)
                            self.fruit_labels.append(self.fruit_classes.index(fruit_class))

                            # Standardize ripeness class naming
                            standardized_ripeness_class = ripeness_class.capitalize()
                            if standardized_ripeness_class in self.ripeness_classes:
                                self.ripeness_labels.append(self.ripeness_classes.index(standardized_ripeness_class))
                            else:
                                raise ValueError(f"Unknown ripeness class: {ripeness_class}")
        
    def __len__(self):
        return len(self.image_paths)  # Returns the total number of images
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Open image and convert to RGB
        fruit_label = self.fruit_labels[idx]  # Get corresponding fruit label
        ripeness_label = self.ripeness_labels[idx]  # Get corresponding ripeness label
        
        if self.transform:
            image = self.transform(image)  # Apply transformations (e.g., resizing, normalization)
        
        return image, fruit_label, ripeness_label  # Return the processed image and labels

# Data transformations (resize, flip, normalize)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 pixels
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally (data augmentation)
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with mean and std dev
])

# Load the dataset
dataset = FruitRipenessDataset(root_dir=data_dir, transform=data_transforms)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders to handle batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Training data loader
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # Validation data loader

# Define a simple CNN model from scratch
class SimpleCNN(nn.Module):
    def __init__(self, num_fruit_classes, num_ripeness_classes):
        super(SimpleCNN, self).__init__()
        # First convolutional layer: input channels=3 (RGB), output channels=32, kernel size=3x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Second convolutional layer: input channels=32, output channels=64, kernel size=3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Third convolutional layer: input channels=64, output channels=128, kernel size=3x3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer to down-sample the image size
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Fully connected layer
        self.fc2_fruit = nn.Linear(512, num_fruit_classes)  # Output layer for fruit classification
        self.fc2_ripeness = nn.Linear(512, num_ripeness_classes)  # Output layer for ripeness classification
        self.dropout = nn.Dropout(0.5)  # Dropout layer to prevent overfitting

    def forward(self, x):
        # Forward pass through the network
        x = self.pool(nn.functional.relu(self.conv1(x)))  # Pass through conv1, ReLU activation, and pooling
        x = self.pool(nn.functional.relu(self.conv2(x)))  # Pass through conv2, ReLU activation, and pooling
        x = self.pool(nn.functional.relu(self.conv3(x)))  # Pass through conv3, ReLU activation, and pooling
        x = x.view(-1, 128 * 28 * 28)  # Flatten the feature maps
        x = self.dropout(nn.functional.relu(self.fc1(x)))  # Pass through fc1, ReLU activation, and dropout
        fruit_output = self.fc2_fruit(x)  # Output for fruit classification
        ripeness_output = self.fc2_ripeness(x)  # Output for ripeness classification
        return fruit_output, ripeness_output  # Return both outputs

# Initialize the model
num_fruit_classes = len(dataset.fruit_classes)  # Number of fruit classes in the dataset
num_ripeness_classes = len(dataset.ripeness_classes)  # Number of ripeness classes
model = SimpleCNN(num_fruit_classes=num_fruit_classes, num_ripeness_classes=num_ripeness_classes)
model = model.to(device)  # Move the model to the GPU if available

# Debug check to confirm model is on GPU
print(f"Model is on device: {next(model.parameters()).device}")

# Loss functions for fruit and ripeness classification
fruit_criterion = nn.CrossEntropyLoss()
ripeness_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer (Adam) to update model weights

# Early stopping parameters to avoid overfitting
patience = 10  # Number of batches to wait before stopping if no improvement
min_delta = 0.001  # Minimum change to consider as improvement
best_loss = float('inf')  # Track the best loss
batches_without_improvement = 0  # Count batches without improvement

# Training loop with detailed debug prints and early stopping
num_epochs = 10
print("Starting training...")

train_roc_data = {'labels': [], 'probs': []}  # Store true labels and probs for ROC during training

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0  # Track running loss
    fruit_correct = 0  # Track correct fruit classifications
    ripeness_correct = 0  # Track correct ripeness classifications
    total = 0  # Track total samples

    print(f"Epoch {epoch+1}/{num_epochs}...")
    for batch_idx, (images, fruit_labels, ripeness_labels) in enumerate(train_loader):
        images = images.to(device)  # Move images to device (GPU/CPU)
        fruit_labels = fruit_labels.to(device)  # Move fruit labels to device
        ripeness_labels = ripeness_labels.to(device)  # Move ripeness labels to device
        
        optimizer.zero_grad()  # Zero the parameter gradients
        fruit_outputs, ripeness_outputs = model(images)  # Forward pass
        fruit_loss = fruit_criterion(fruit_outputs, fruit_labels)  # Calculate loss for fruit classification
        ripeness_loss = ripeness_criterion(ripeness_outputs, ripeness_labels)  # Calculate loss for ripeness classification
        loss = fruit_loss + ripeness_loss  # Total loss
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update weights

        running_loss += loss.item() * images.size(0)  # Update running loss
        _, fruit_preds = torch.max(fruit_outputs, 1)  # Get predicted fruit classes
        _, ripeness_preds = torch.max(ripeness_outputs, 1)  # Get predicted ripeness classes
        total += fruit_labels.size(0)  # Update total samples
        fruit_correct += (fruit_preds == fruit_labels).sum().item()  # Count correct fruit predictions
        ripeness_correct += (ripeness_preds == ripeness_labels).sum().item()  # Count correct ripeness predictions

        # Store probabilities and true labels for ROC curve
        train_roc_data['probs'].extend(nn.functional.softmax(ripeness_outputs, dim=1).detach().cpu().numpy())
        train_roc_data['labels'].extend(ripeness_labels.cpu().numpy())

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Early stopping check
        if loss.item() < best_loss - min_delta:
            best_loss = loss.item()
            batches_without_improvement = 0
            print(f"New best loss: {best_loss:.4f} at batch {batch_idx+1}")
        else:
            batches_without_improvement += 1
            if batches_without_improvement >= patience:
                print(f"Stopping early at epoch {epoch+1}, batch {batch_idx+1} due to no improvement.")
                break

    if batches_without_improvement >= patience:
        break

    epoch_loss = running_loss / len(train_loader.dataset)  # Average loss for the epoch
    fruit_accuracy = fruit_correct / total  # Fruit classification accuracy
    ripeness_accuracy = ripeness_correct / total  # Ripeness classification accuracy

    print(f"End of Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Fruit Accuracy: {fruit_accuracy:.4f}, Ripeness Accuracy: {ripeness_accuracy:.4f}")

# Print the final loss score after training
print(f"Final Loss Score after Training: {epoch_loss:.4f}")

print("Training complete.")

# Save the model
torch.save(model.state_dict(), 'fruit_ripeness_model.pth')

# Plot ROC curve for training phase
train_probs = np.array(train_roc_data['probs'])
train_labels = np.array(train_roc_data['labels'])

for i in range(num_ripeness_classes):
    y_true = (train_labels == i).astype(int)
    y_score = train_probs[:, i]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    plt.plot(fpr, tpr, label=f'Training - {dataset.ripeness_classes[i]} (AUC = {auc_score:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Training Phase (Ripeness)')
plt.legend(loc="lower right")
plt.show()

# Validation loop and performance evaluation
print("Starting validation...")
model.eval()  # Set model to evaluation mode
val_fruit_preds = []
val_ripeness_preds = []
val_fruit_labels = []
val_ripeness_labels = []
val_probs = []

with torch.no_grad():  # Disable gradient calculation for validation
    for images, fruit_labels, ripeness_labels in val_loader:
        images = images.to(device)
        fruit_labels = fruit_labels.to(device)
        ripeness_labels = ripeness_labels.to(device)
        
        fruit_outputs, ripeness_outputs = model(images)  # Forward pass
        _, fruit_preds = torch.max(fruit_outputs, 1)  # Get predicted fruit classes
        _, ripeness_preds = torch.max(ripeness_outputs, 1)  # Get predicted ripeness classes

        val_fruit_preds.extend(fruit_preds.cpu().numpy())  # Store predicted fruit classes
        val_ripeness_preds.extend(ripeness_preds.cpu().numpy())  # Store predicted ripeness classes
        val_fruit_labels.extend(fruit_labels.cpu().numpy())  # Store true fruit labels
        val_ripeness_labels.extend(ripeness_labels.cpu().numpy())  # Store true ripeness labels
        val_probs.extend(nn.functional.softmax(ripeness_outputs, dim=1).cpu().numpy())  # Store ripeness probabilities

# Convert lists to numpy arrays for evaluation
val_fruit_preds = np.array(val_fruit_preds)
val_ripeness_preds = np.array(val_ripeness_preds)
val_fruit_labels = np.array(val_fruit_labels)
val_ripeness_labels = np.array(val_ripeness_labels)
val_probs = np.array(val_probs)

# Calculate accuracy, precision, recall, F1-score
fruit_accuracy = accuracy_score(val_fruit_labels, val_fruit_preds)
ripeness_accuracy = accuracy_score(val_ripeness_labels, val_ripeness_preds)

fruit_precision, fruit_recall, fruit_f1, _ = precision_recall_fscore_support(val_fruit_labels, val_fruit_preds, average='weighted')
ripeness_precision, ripeness_recall, ripeness_f1, _ = precision_recall_fscore_support(val_ripeness_labels, val_ripeness_preds, average='weighted')

print(f'Fruit - Accuracy: {fruit_accuracy:.4f}, Precision: {fruit_precision:.4f}, Recall: {fruit_recall:.4f}, F1-score: {fruit_f1:.4f}')
print(f'Ripeness - Accuracy: {ripeness_accuracy:.4f}, Precision: {ripeness_precision:.4f}, Recall: {ripeness_recall:.4f}, F1-score: {ripeness_f1:.4f}')

print("Validation complete.")

# Plot ROC curve for validation phase
for i in range(num_ripeness_classes):
    y_true = (val_ripeness_labels == i).astype(int)
    y_score = val_probs[:, i]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    plt.plot(fpr, tpr, label=f'Validation - {dataset.ripeness_classes[i]} (AUC = {auc_score:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Validation Phase (Ripeness)')
plt.legend(loc="lower right")
plt.show()
