import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report, cohen_kappa_score, matthews_corrcoef, log_loss
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Debug: Starting the script
print("Starting the script...")

# Define the data classes (ensure all folder names match exactly)
fruit_classes = ['BananaDB', 'CucumberQ', 'GrapeQ', 'KakiQ', 'PapayaQ', 'PeachQ', 'PearQ', 'PepperQ', 'StrawberryQ', 'tomatoQ', 'WatermeloQ']
ripeness_classes = ['Fresh', 'Mild', 'Rotten']

# Debug: Displaying the fruit and ripeness classes
print(f"Fruit classes: {fruit_classes}")
print(f"Ripeness classes: {ripeness_classes}")

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Debug: Transformation pipeline created
print("Transformation pipeline created.")

# Define the custom dataset
class FruitRipenessDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.fruit_labels = []
        self.ripeness_labels = []
        self.fruit_classes = fruit_classes
        self.ripeness_classes = ripeness_classes

        # Populate the dataset
        print("Loading dataset from:", root_dir)
        for fruit_class in os.listdir(root_dir):
            fruit_path = os.path.join(root_dir, fruit_class)
            if os.path.isdir(fruit_path):
                print(f"Found fruit class directory: {fruit_class}")
                for ripeness_class in os.listdir(fruit_path):
                    ripeness_path = os.path.join(fruit_path, ripeness_class)
                    if os.path.isdir(ripeness_path):
                        print(f"  Found ripeness class directory: {ripeness_class}")
                        for img_name in os.listdir(ripeness_path):
                            img_path = os.path.join(ripeness_path, img_name)
                            self.image_paths.append(img_path)
                            try:
                                fruit_label_index = self.fruit_classes.index(fruit_class)
                                ripeness_label_index = self.ripeness_classes.index(ripeness_class.capitalize())
                                self.fruit_labels.append(fruit_label_index)
                                self.ripeness_labels.append(ripeness_label_index)
                                print(f"    Loaded image: {img_name}, Fruit: {fruit_class}, Ripeness: {ripeness_class}")
                            except ValueError as e:
                                print(f"Error: {e} for image {img_name}")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        fruit_label = self.fruit_labels[idx]
        ripeness_label = self.ripeness_labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, fruit_label, ripeness_label

# Debug: Dataset class defined
print("Dataset class defined.")

# Load the trained model (replace with the correct path to the saved model)
class SimpleCNN(nn.Module):
    def __init__(self, num_fruit_classes, num_ripeness_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2_fruit = nn.Linear(512, num_fruit_classes)
        self.fc2_ripeness = nn.Linear(512, num_ripeness_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        fruit_output = self.fc2_fruit(x)
        ripeness_output = self.fc2_ripeness(x)
        return fruit_output, ripeness_output

# Debug: Model class defined
print("Model class defined.")

# Load the trained model
num_fruit_classes = len(fruit_classes)
num_ripeness_classes = len(ripeness_classes)
model = SimpleCNN(num_fruit_classes=num_fruit_classes, num_ripeness_classes=num_ripeness_classes)
model.load_state_dict(torch.load('fruit_ripeness_model.pth', map_location=torch.device('cpu')))
model.eval()
print("Model loaded and set to evaluation mode.")

# Load the validation dataset
data_dir = 'D:/PUK/Honiers/AI/SEM2/Sem_CNN/Fruq-multi'
print(f"Loading dataset from {data_dir}...")
dataset = FruitRipenessDataset(root_dir=data_dir, transform=transform)
print(f"Dataset loaded. Total images: {len(dataset)}")

# Use a DataLoader for batch processing
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
print("DataLoader created.")

# Prepare lists to store actual and predicted labels
all_fruit_labels = []
all_ripeness_labels = []
all_fruit_preds = []
all_ripeness_preds = []
all_fruit_probs = []  # Store predicted probabilities for ROC

# Evaluation loop
print("Starting evaluation loop...")
with torch.no_grad():
    for batch_idx, (images, fruit_labels, ripeness_labels) in enumerate(val_loader):
        print(f"Evaluating batch {batch_idx+1}/{len(val_loader)}...")
        fruit_outputs, ripeness_outputs = model(images)
        _, fruit_preds = torch.max(fruit_outputs, 1)
        _, ripeness_preds = torch.max(ripeness_outputs, 1)

        all_fruit_labels.extend(fruit_labels.cpu().numpy())
        all_ripeness_labels.extend(ripeness_labels.cpu().numpy())
        all_fruit_preds.extend(fruit_preds.cpu().numpy())
        all_ripeness_preds.extend(ripeness_preds.cpu().numpy())
        all_fruit_probs.extend(fruit_outputs.cpu().numpy())

print("Evaluation loop completed.")

# Convert lists to numpy arrays for evaluation
all_fruit_labels = np.array(all_fruit_labels)
all_ripeness_labels = np.array(all_ripeness_labels)
all_fruit_preds = np.array(all_fruit_preds)
all_ripeness_preds = np.array(all_ripeness_preds)
all_fruit_probs = np.array(all_fruit_probs)

# Calculate accuracy, precision, recall, F1-score for both fruit type and ripeness
print("Calculating performance metrics...")
fruit_accuracy = accuracy_score(all_fruit_labels, all_fruit_preds)
ripeness_accuracy = accuracy_score(all_ripeness_labels, all_ripeness_preds)

fruit_precision, fruit_recall, fruit_f1, _ = precision_recall_fscore_support(all_fruit_labels, all_fruit_preds, average='weighted')
ripeness_precision, ripeness_recall, ripeness_f1, _ = precision_recall_fscore_support(all_ripeness_labels, all_ripeness_preds, average='weighted')

print(f'Fruit - Accuracy: {fruit_accuracy:.4f}, Precision: {fruit_precision:.4f}, Recall: {fruit_recall:.4f}, F1-score: {fruit_f1:.4f}')
print(f'Ripeness - Accuracy: {ripeness_accuracy:.4f}, Precision: {ripeness_precision:.4f}, Recall: {ripeness_recall:.4f}, F1-score: {ripeness_f1:.4f}')

# Plot ROC curves for multi-class fruit classification (one-vs-rest approach)
print("Calculating ROC AUC for multi-class (one-vs-rest)...")
for i in range(num_fruit_classes):
    y_true = (all_fruit_labels == i).astype(int)  # Binarize the labels for class `i`
    y_score = all_fruit_probs[:, i]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    plt.plot(fpr, tpr, label=f'{fruit_classes[i]} (AUC = {auc_score:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-Class Fruit Classification (One-vs-Rest)')
plt.legend(loc="lower right")
plt.show()

print("Script execution completed.")
