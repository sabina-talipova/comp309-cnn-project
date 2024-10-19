import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train import CustomCNN

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total}%')

model = CustomCNN()
model.load_state_dict(torch.load('model.pth'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Случайное горизонтальное отражение
    transforms.RandomRotation(20),  # Случайное вращение
    transforms.RandomResizedCrop(224),  # Случайное обрезание и изменение размера
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Resize((224, 224)),  # Изменяем размер
    transforms.ToTensor(),  # Преобразуем изображения в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
])

test_loader = datasets.ImageFolder(root='./data', transform=train_transform)
test_loader = DataLoader(test_loader, batch_size=32, shuffle=True)

test_model(model, test_loader)