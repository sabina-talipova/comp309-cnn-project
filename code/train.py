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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from dotenv import load_dotenv

DATA_FOLDER_PATH = os.environ['FILE_PATH']
MODEL_FILE_PATH = 'model.pth'

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers (for 224x224 images)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # For images 224x224
        self.fc2 = nn.Linear(128, 3)  # 3 classes

        # ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Direct propagation through layers
        x = self.pool(self.relu(self.conv1(x)))  # Convolutional layer -> ReLU -> Max Pooling
        x = self.pool(self.relu(self.conv2(x)))  # Convolutional layer -> ReLU -> Max Pooling
        x = x.view(-1, 32 * 56 * 56)  # Reshape to match fully connected layers
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


def is_valid_file(file_path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    if "ipynb" in file_path:
      return False
    return True

class ModelTrainer:
    def __init__(self, dir):
        self.dir = dir
        self.__model = CustomCNN()
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
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # self.remove_wrong_folders()

        train_dataset = datasets.ImageFolder(root=self.dir, transform=train_transform, is_valid_file=is_valid_file)
        self.__train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=4)

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
    my_object = ModelTrainer(DATA_FOLDER_PATH)
    my_object.create_model()
    print("Training model DONE")

if __name__ == "__main__":
    main()
