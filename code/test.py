import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train import CustomCNN


def test_model(model, test_loader):
    """
    Evaluates the performance of the given model on the provided test data.

    Args:
        model: The neural network model to be tested.
        test_loader: DataLoader containing the test dataset.

    Prints:
        The accuracy of the model on the test images as a percentage.
    """
    # Set the model to evaluation mode, which disables dropout and batch normalization
    model.eval()

    correct = 0  # Counter for correct predictions
    total = 0  # Total number of samples processed

    # Disable gradient calculation to speed up the evaluation process
    with torch.no_grad():
        # Iterate through the test dataset
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device (GPU/CPU)
            outputs = model(inputs)  # Forward pass through the model

            # Get the predicted class by finding the index of the maximum output value
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)  # Update the total sample count
            correct += (predicted == labels).sum().item()  # Count correct predictions

    # Calculate and print the accuracy of the model on the test set
    print(f'Accuracy of the model on the test images: {100 * correct / total}%')


# Initialize the model
model = CustomCNN()

# Load the model's state_dict from the saved best model file
model.load_state_dict(torch.load('model.pth')['model_state_dict'])

# Set the device to GPU if available; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the appropriate device
model = model.to(device)

# Define transformations to apply to the test images
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
    transforms.RandomRotation(20),  # Randomly rotate images
    transforms.RandomResizedCrop(224),  # Randomly crop and resize to 224x224
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color adjustments
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize tensor
])

# Create a DataLoader for the test dataset
test_loader = datasets.ImageFolder(root='testdata', transform=train_transform)
test_loader = DataLoader(test_loader, batch_size=32, shuffle=True)  # Load the test data in batches of 32

# Call the function to test the model on the test dataset
test_model(model, test_loader)
