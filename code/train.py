import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from dotenv import load_dotenv

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # Сверточный слой
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Слой подвыборки (Max Pooling)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Полносвязные слои
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Для изображений размером 32x32 (например, CIFAR-10)
        self.fc2 = nn.Linear(128, 10)  # Предположим, что у нас 10 классов для классификации

        # Функция активации ReLU
        self.relu = nn.ReLU()

    def forward(self, x):
        # Прямое распространение через слои
        x = self.pool(self.relu(self.conv1(x)))  # Свертка -> ReLU -> Max Pooling
        x = self.pool(self.relu(self.conv2(x)))  # Свертка -> ReLU -> Max Pooling
        x = x.view(-1, 32 * 8 * 8)  # Выравнивание для полносвязного слоя
        x = self.relu(self.fc1(x))  # Полносвязный слой -> ReLU
        x = self.fc2(x)  # Полносвязный слой -> выход

        return x

class ModelTrainer:
    def __init__(self, dir):
        self.dir = dir
        self.__model = CustomCNN()
        self.__train_loader = None
        self.__criterion = None
        self.__optimizer = None
        self.__device = None

    def load_data(self):
        print("Start load data...")

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # Случайное горизонтальное отражение
            transforms.RandomRotation(20),  # Случайное вращение
            transforms.RandomResizedCrop(224),  # Случайное обрезание и изменение размера
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize((224, 224)),  # Изменяем размер
            transforms.ToTensor(),  # Преобразуем изображения в тензор
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
        ])

        train_dataset = datasets.ImageFolder(root='./data', transform=train_transform)
        self.__train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        print("Load data DONE")

    def create_model(self):
        print("Start model creation...")

        # Устанавливаем устройство для вычислений (GPU, если доступно)
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model = self.__model.to(self.__device)

        # Определяем функцию потерь и оптимизатор
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = optim.Adam(self.__model.parameters(), lr=0.001)

        inputs = torch.randn(10)
        output = self.__model(inputs)

        torch.save({
            'model_state_dict': self.__model.state_dict(),
            'optimizer_state_dict': self.__optimizer.state_dict(),
        }, 'model.pth')

        print("Model was saved in model.pth")

    def train_model(self, num_epochs=10):
        print("Start model training...")

        self.__model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in self.__train_loader:
                inputs, labels = inputs.to(self.__device), labels.to(self.__device)

                # Обнуляем градиенты
                self.__optimizer.zero_grad()

                # Прямое распространение
                outputs = self.__model(inputs)
                loss = self.__criterion(outputs, labels)

                # Обратное распространение и оптимизация
                loss.backward()
                self.__optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(self.__train_loader)}")

        print("Model training DONE")

def main():
    load_dotenv()
    print("Training model...")
    my_object = ModelTrainer(os.environ['FILE_PATH'])

    # Вызываем метод greet
    my_object.create_model()
    my_object.train_model(num_epochs=10)
    print("Training model DONE")

if __name__ == "__main__":
    main()
