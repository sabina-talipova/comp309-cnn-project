import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from dotenv import load_dotenv
from tqdm import tqdm

from sklearn.model_selection import KFold

DATA_FOLDER_PATH = "train_data"
# DATA_FOLDER_PATH = os.environ['FILE_PATH']
MODEL_FILE_PATH = 'model.pth'

INPUT_SIZE = 224   # Image 224x224 and 3 channels
HIDDEN_SIZE = 128  # Features in hidden layer
NUM_CLASSES = 3    # Output
BATCH_SIZE = 32

class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for image classification.

    This model consists of two fully connected layers with a ReLU activation in between.
    - Input layer (fc1): Transforms a flattened image input of size 224x224x3 into 128 features.
    - Output layer (fc2): Maps the 128 features to the 3 output classes.

    The model is designed for use with 224x224 RGB images, which are flattened before
    passing through the linear layers.

    Methods:
        forward(x): Defines the forward pass of the network.
    """
    def __init__(self):
        super(SimpleMLP, self).__init__()

        input_size = INPUT_SIZE * INPUT_SIZE * NUM_CLASSES

        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.input_size = int((INPUT_SIZE / 4) * (INPUT_SIZE / 4) * BATCH_SIZE)

        # Fully connected layers (for 224x224 images)
        self.fc1 = nn.Linear(self.input_size, HIDDEN_SIZE)  # For images 224x224
        self.fc2 = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)  # 3 classes

        # ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Direct propagation through layers
        x = self.pool(self.relu(self.conv1(x)))  # Convolutional layer -> ReLU -> Max Pooling
        x = self.pool(self.relu(self.conv2(x)))  # Convolutional layer -> ReLU -> Max Pooling
        x = x.view(-1, self.input_size)  # Reshape to match fully connected layers
        x = self.relu(self.fc1(x))  # Fully connected layers -> ReLU
        x = self.fc2(x)  # Fully connected layers -> output

        return x

class ModelEDA:
  @staticmethod
  def load_images_from_folder(folder):
      images = []
      for root, dirs, files in os.walk(folder):
          for filename in files:
              file_path = os.path.join(root, filename)
              if filename.endswith(('.png', '.jpg', '.jpeg')):
                  img = cv2.imread(file_path)
                  if img is not None:
                      images.append(img)
      return images

  @staticmethod
  def show_images(images, num_images=5):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.show()

  @staticmethod
  def analyze_image_sizes(images):
    sizes = [img.shape[:2] for img in images]
    unique_sizes = set(sizes)

    print(f"Unique image sizes: {unique_sizes}")

    sizes_list = [str(size) for size in sizes]
    sns.countplot(y=sizes_list)
    plt.title("Distribution of Image Sizes")
    plt.show()

  @staticmethod
  def analyze_color_channels(images):
    image = images[0]

    channels = ['Blue', 'Green', 'Red']
    for i, channel in enumerate(channels):
        plt.hist(image[:,:,i].ravel(), bins=256, color=channel.lower(), alpha=0.5)
        plt.title(f"{channel} Channel Distribution")
        plt.show()

  @staticmethod
  def analyze_class_distribution(image_labels):
    sns.countplot(image_labels)
    plt.title("Class Distribution")
    plt.show()

  @staticmethod
  def analyze_brightness(images):
    brightness = []
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness.append(np.mean(hsv[:,:,2]))

    plt.hist(brightness, bins=50)
    plt.title("Brightness Distribution")
    plt.show()

  @staticmethod
  def check_for_duplicates(images):
    unique_images = set([img.tobytes() for img in images])
    num_duplicates = len(images) - len(unique_images)
    print(f"Number of duplicate images: {num_duplicates}")


class ModelTrainer:
    def __init__(self, dir, model):
        self.dir = dir
        self.__model = model
        self.__train_loader = None
        self.__criterion = None
        self.__optimizer = None
        self.__device = None

    def remove_wrong_folders(self):
      for root, dirs, files in os.walk(self.dir):
        for dir_name in dirs:
            if dir_name == ".ipynb_checkpoints":
                dir_path = os.path.join(root, dir_name)
                shutil.rmtree(dir_path)
                print(f"Deleted: {dir_path}")

    def load_data(self):
        print("Start load data...")

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(INPUT_SIZE),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # self.remove_wrong_folders()

        def is_valid_file(file_path):
            valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
            if "ipynb" in file_path:
                return False
            return True

        train_dataset = datasets.ImageFolder(root=self.dir, transform=train_transform, is_valid_file=is_valid_file)
        self.__train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        print("Load data DONE")

    def create_model(self):
        print("Start model creation...")

        if self.__model is None:
            raise ValueError("Model is not initialized. Please initialize the model before calling create_model.")

        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__model = self.__model.to(self.__device)

        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = optim.Adam(self.__model.parameters(), lr=0.001)
        self.load_data()
        self.train_model(num_epochs=10)

        try:
            torch.save({
                'model_state_dict': self.__model.state_dict(),
                'optimizer_state_dict': self.__optimizer.state_dict(),
            }, MODEL_FILE_PATH)
            print("Model was saved in model.pth")
        except Exception as e:
            print(f"Error saving the model: {e}")


    def train_model(self, num_epochs=10):
        print("Start model training...")

        self.__model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0

            if len(self.__train_loader) == 0:
                print("No data available in train_loader.")
                return

            for batch_index, (inputs, labels) in enumerate(self.__train_loader):
                inputs, labels = inputs.to(self.__device), labels.to(self.__device)

                self.__optimizer.zero_grad()

                outputs = self.__model(inputs)
                loss = self.__criterion(outputs, labels)

                loss.backward()
                self.__optimizer.step()

                running_loss += loss.item()

                if batch_index % 10 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_index}, Loss: {loss.item()}")

            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {running_loss / len(self.__train_loader)}")

        print("Model training DONE")

    def train_model_CV(self, num_epochs=10):
        print("Start Cross-validation model training...")

        BATCH_SIZE = 32
        NUM_EPOCHS = 5
        NUM_FOLDS = 5

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(INPUT_SIZE),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        def is_valid_file(file_path):
            valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
            if "ipynb" in file_path:
                return False
            return True

        dataset = datasets.ImageFolder(root=self.dir, transform=train_transform, is_valid_file=is_valid_file)

        kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)


        criterion = nn.CrossEntropyLoss()

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print(f'Fold {fold + 1}/{NUM_FOLDS}')

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

            model = self.__model
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            for epoch in range(NUM_EPOCHS):

                model.train()
                running_loss = 0.0
                for inputs, labels in tqdm(train_loader, desc=f"Training Fold {fold + 1}, Epoch {epoch + 1}"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                print(f"Fold {fold + 1}, Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}")

                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print(f"Fold {fold + 1}, Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}, "
                      f"Accuracy: {100 * correct / total}%")

        print("Model training DONE")

        try:
            torch.save({
                'model_state_dict': self.__model.state_dict(),
                'optimizer_state_dict': self.__optimizer.state_dict(),
            }, MODEL_FILE_PATH)
            print("Model was saved in model.pth")
        except Exception as e:
            print(f"Error saving the model: {e}")


def eda():
    load_dotenv()
    print("Training model...")

    modelEDA = ModelEDA()
    images = modelEDA.load_images_from_folder(DATA_FOLDER_PATH)

    modelEDA.show_images(images)
    modelEDA.analyze_image_sizes(images)
    modelEDA.analyze_color_channels(images)

    image_labels = ['tomato', 'strawberry', 'cherry']

    modelEDA.analyze_class_distribution(image_labels)
    modelEDA.analyze_brightness(images)
    modelEDA.check_for_duplicates(images)

def main():
    load_dotenv()
    print("Training model...")
    my_object = ModelTrainer(DATA_FOLDER_PATH, CustomCNN())
    my_object.train_model_CV()
    # my_object.create_model()
    print("Training model DONE")

if __name__ == "__main__":
    main()
