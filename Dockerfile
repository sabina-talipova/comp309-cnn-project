FROM python:3.9-slim

RUN apt-get update && apt-get install -y cron bash nano && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install Flask

CMD ["python", "app.py"]
