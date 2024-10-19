# Python image
FROM python:3.9-slim

# Установите cron, bash и nano
RUN apt-get update && apt-get install -y cron bash nano && apt-get clean && rm -rf /var/lib/apt/lists/*

# Установите рабочую директорию
WORKDIR /app

# Копируйте файлы проекта в контейнер
COPY . .

# Установите зависимости
RUN pip install Flask

# Запустите ваш скрипт (если необходимо)
CMD ["python", "app.py"]
