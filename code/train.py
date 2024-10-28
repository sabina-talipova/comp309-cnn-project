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

        # Convolutional layer 1: Takes an input with 3 channels (e.g., RGB image),
        # applies 16 filters of size 3x3 with padding of 1 to retain spatial dimensions
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

        # Convolutional layer 2: Takes the output of conv1 (16 channels),
        # applies 32 filters of size 3x3 with padding of 1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Max Pooling layer: Reduces the spatial dimensions of the feature maps by half (kernel size 2x2, stride 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the flattened size for the fully connected layer input,
        # assuming the input image has dimensions (INPUT_SIZE x INPUT_SIZE) and is downsampled twice by pooling layers
        self.input_size = int((INPUT_SIZE / 4) * (INPUT_SIZE / 4) * BATCH_SIZE)

        # Fully connected layer 1: Maps the flattened features to a hidden layer of size HIDDEN_SIZE
        self.fc1 = nn.Linear(self.input_size, HIDDEN_SIZE)  # For images 224x224

        # Fully connected layer 2 (Output layer): Maps to NUM_CLASSES, providing class scores for classification
        self.fc2 = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)  # 3 classes

        # ReLU activation function, to introduce non-linearity
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass input through first convolutional layer, apply ReLU activation, then max pooling
        x = self.pool(self.relu(self.conv1(x)))

        # Pass through second convolutional layer, apply ReLU activation, then max pooling
        x = self.pool(self.relu(self.conv2(x)))

        # Flatten feature maps to prepare for fully connected layers
        x = x.view(-1, self.input_size)

        # Pass through the first fully connected layer with ReLU activation
        x = self.relu(self.fc1(x))

        # Pass through the final fully connected layer to output class scores
        x = self.fc2(x)

        return x


class ModelEDA:
    @staticmethod
    def load_images_from_folder(folder):
        # Load all images from a given folder path
        images = []
        # Traverse through all files and subdirectories in the specified folder
        for root, dirs, files in os.walk(folder):
            for filename in files:
                # Construct the file path
                file_path = os.path.join(root, filename)
                # Check if the file is an image by its extension
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    # Read the image using OpenCV
                    img = cv2.imread(file_path)
                    if img is not None:
                        images.append(img)
        return images

    @staticmethod
    def show_images(images, num_images=5):
        # Display a specified number of images
        plt.figure(figsize=(10, 10))
        for i in range(num_images):
            # Create subplot for each image
            plt.subplot(1, num_images, i+1)
            # Convert image color from BGR to RGB and display
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            plt.axis('off')  # Hide axes
        plt.show()

    @staticmethod
    def analyze_image_sizes(images):
        # Analyze the dimensions (height, width) of each image in the list
        sizes = [img.shape[:2] for img in images]  # Get (height, width) for each image
        unique_sizes = set(sizes)  # Find unique image sizes

        print(f"Unique image sizes: {unique_sizes}")  # Output unique sizes

        # Plot the distribution of image sizes
        sizes_list = [str(size) for size in sizes]
        sns.countplot(y=sizes_list)
        plt.title("Distribution of Image Sizes")
        plt.show()

    @staticmethod
    def analyze_color_channels(images):
        # Analyze color channel distribution for the first image in the list
        image = images[0]  # Select the first image

        # Define color channel names
        channels = ['Blue', 'Green', 'Red']
        for i, channel in enumerate(channels):
            # Plot histogram for each color channel
            plt.hist(image[:, :, i].ravel(), bins=256, color=channel.lower(), alpha=0.5)
            plt.title(f"{channel} Channel Distribution")
            plt.show()

    @staticmethod
    def analyze_class_distribution(image_labels):
        # Analyze and display class distribution using the image labels provided
        sns.countplot(image_labels)
        plt.title("Class Distribution")
        plt.show()

    @staticmethod
    def analyze_brightness(images):
        # Analyze the brightness of each image by converting to HSV color space
        brightness = []
        for img in images:
            # Convert image to HSV, then calculate the mean brightness (V channel)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            brightness.append(np.mean(hsv[:, :, 2]))

        # Plot the distribution of brightness levels
        plt.hist(brightness, bins=50)
        plt.title("Brightness Distribution")
        plt.show()

    @staticmethod
    def check_for_duplicates(images):
        # Check for duplicate images by converting each to a byte representation
        unique_images = set([img.tobytes() for img in images])
        num_duplicates = len(images) - len(unique_images)  # Calculate the number of duplicates
        print(f"Number of duplicate images: {num_duplicates}")

class ModelTrainer:
    def __init__(self, dir, model):
        """
        Initializes the ModelTrainer instance.
        Args:
            dir (str): Directory containing image data.
            model (torch.nn.Module): PyTorch model to be trained.
        """
        self.dir = dir
        self.__model = model
        self.__train_loader = None  # DataLoader for training data
        self.__criterion = None  # Loss function
        self.__optimizer = None  # Optimizer for model training
        self.__device = None  # Device on which to train the model (CPU or GPU)

    def remove_wrong_folders(self):
        """
        Removes unnecessary folders (e.g., '.ipynb_checkpoints') in the data directory to avoid errors.
        """
        for root, dirs, files in os.walk(self.dir):
            for dir_name in dirs:
                if dir_name == ".ipynb_checkpoints":
                    dir_path = os.path.join(root, dir_name)
                    shutil.rmtree(dir_path)
                    print(f"Deleted: {dir_path}")

    def load_data(self):
        """
        Loads and transforms the dataset, then initializes the DataLoader for training.
        Applies various image transformations for data augmentation.
        """
        print("Start load data...")

        # Define transformations for data augmentation and preprocessing
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(INPUT_SIZE),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Helper function to filter valid files
        def is_valid_file(file_path):
            valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
            return not "ipynb" in file_path

        # Load the dataset with transformations and filtering
        train_dataset = datasets.ImageFolder(root=self.dir, transform=train_transform, is_valid_file=is_valid_file)
        self.__train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        print("Load data DONE")

    def create_model(self):
        """
        Configures the model, criterion, and optimizer. Loads data and starts the training process.
        Also saves the model checkpoint after training.
        """
        print("Start model creation...")

        # Check if the model is initialized
        if self.__model is None:
            raise ValueError("Model is not initialized. Please initialize the model before calling create_model.")

        # Set the device to GPU if available, otherwise CPU
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move the model to the chosen device
        self.__model = self.__model.to(self.__device)

        # Define the loss function and optimizer
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = optim.Adam(self.__model.parameters(), lr=0.001)

        # Load the data and start training
        self.load_data()
        self.train_model(num_epochs=10)

        # Save model and optimizer state after training
        try:
            torch.save({
                'model_state_dict': self.__model.state_dict(),
                'optimizer_state_dict': self.__optimizer.state_dict(),
            }, MODEL_FILE_PATH)
            print("Model was saved in model.pth")
        except Exception as e:
            print(f"Error saving the model: {e}")

    def train_model(self, num_epochs=10):
        """
        Trains the model for a specified number of epochs using the training DataLoader.
        Reports the average loss at each epoch and prints batch loss periodically.
        """
        print("Start model training...")

        self.__model.train()  # Set model to training mode
        for epoch in range(num_epochs):
            running_loss = 0.0

            # Check for available data
            if len(self.__train_loader) == 0:
                print("No data available in train_loader.")
                return

            # Iterate over each batch in the DataLoader
            for batch_index, (inputs, labels) in enumerate(self.__train_loader):
                inputs, labels = inputs.to(self.__device), labels.to(self.__device)

                # Zero the parameter gradients
                self.__optimizer.zero_grad()

                # Forward pass
                outputs = self.__model(inputs)
                loss = self.__criterion(outputs, labels)

                # Backward pass and optimization step
                loss.backward()
                self.__optimizer.step()

                running_loss += loss.item()

                # Print loss for every 10th batch
                if batch_index % 10 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_index}, Loss: {loss.item()}")

            # Print the average loss for each epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {running_loss / len(self.__train_loader)}")

        print("Model training DONE")

    def train_model_CV(self, num_epochs=10):
        """
        Implements cross-validation for the model training process.
        Trains and validates the model on multiple folds and saves the best model based on accuracy.
        """
        print("Start Cross-validation model training...")

        # Define cross-validation settings
        BATCH_SIZE = 32
        NUM_EPOCHS = 5
        NUM_FOLDS = 5

        # Define transformations
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(INPUT_SIZE),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Helper function to filter valid files
        def is_valid_file(file_path):
            valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
            return not "ipynb" in file_path

        # Load dataset with transformations and file validation
        dataset = datasets.ImageFolder(root=self.dir, transform=train_transform, is_valid_file=is_valid_file)

        best_model = None  # Track best model based on validation accuracy
        best_accuracy = 0.0

        # Split dataset into k-folds for cross-validation
        kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
        criterion = nn.CrossEntropyLoss()

        # Loop through each fold for training and validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print(f'Fold {fold + 1}/{NUM_FOLDS}')

            # Create train and validation subsets
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

            model = self.__model  # Initialize model
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            # Training and validation within each epoch
            for epoch in range(NUM_EPOCHS):
                # Training phase
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

                # Validation phase
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

                        # Calculate accuracy
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total

                print(f"Fold {fold + 1}, Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}, "
                      f"Accuracy: {accuracy}%")

            # Update best model based on accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

        # Save best model after cross-validation
        if best_model is not None:
            torch.save({
                'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, MODEL_FILE_PATH)
            print("Best model saved as model.pth")
        print("Cross-validation model training DONE")

def eda():
    """
    Perform Exploratory Data Analysis (EDA) on a dataset of images.
    The function initializes the EDA model, loads images from a specified folder,
    and performs various analyses, such as displaying images, analyzing image sizes,
    color channels, class distribution, brightness, and checking for duplicate images.
    """
    # Load environment variables (e.g., for accessing the data folder path)
    load_dotenv()
    print("Training model...")

    # Initialize the ModelEDA class, which contains methods for image analysis
    modelEDA = ModelEDA()

    # Load all images from the specified folder path
    images = modelEDA.load_images_from_folder(DATA_FOLDER_PATH)

    # Display a sample of images to visually inspect the dataset
    modelEDA.show_images(images)

    # Analyze the dimensions and sizes of all images to understand data consistency
    modelEDA.analyze_image_sizes(images)

    # Analyze color channels of images to check for grayscale or RGB channels
    modelEDA.analyze_color_channels(images)

    # Define labels for each image class to analyze their distribution
    image_labels = ['tomato', 'strawberry', 'cherry']

    # Analyze the distribution of each class in the dataset
    modelEDA.analyze_class_distribution(image_labels)

    # Analyze the brightness levels of images to understand overall image quality
    modelEDA.analyze_brightness(images)

    # Check for any duplicate images in the dataset to ensure data quality
    modelEDA.check_for_duplicates(images)


def main():
    """
    Main function to initiate model training.
    Loads environment variables, initializes the model trainer with the data folder path and
    a custom CNN model, and performs cross-validation training.
    """
    # Load environment variables (e.g., for data paths or other configurations)
    load_dotenv()
    print("Training model...")

    # Initialize ModelTrainer with the specified data folder and custom CNN model
    my_object = ModelTrainer(DATA_FOLDER_PATH, CustomCNN())

    # Train the model using cross-validation on the dataset
    my_object.train_model_CV()

    # Optional: Uncomment to train and create the model without cross-validation
    # my_object.create_model()

    print("Training model DONE")

# Entry point to start the main function if this script is run directly
if __name__ == "__main__":
    main()
