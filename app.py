from flask import Flask, request, jsonify
from PIL import Image

# Преобразование изображения для модели
def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image = transform(image).unsqueeze(0)  # Добавляем batch размер
    return image

# Использование модели для предсказания
def predict(image_path):
    model.eval()
    image = preprocess_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Пример использования
label = predict('path_to_your_image.jpg')
print(f'Predicted label: {label}')



app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/predict', methods=['GET'])
def predict():
    # img = request.files['file']
    # img = image.load_img(img, target_size=(224, 224))
    # img_array = image.img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0) / 255.0
    #
    # predictions = model.predict(img_array)
    # predicted_class = np.argmax(predictions)
    #
    # return jsonify({'predicted_class': str(predicted_class)})

    return "Test"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)