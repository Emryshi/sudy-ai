FROM python:3.9-slim

WORKDIR /app

# Sistem bağımlılıklarını kur
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8000

ENV MODEL_PATH=/app/checkpoints/sudy-sft
ENV TOKENIZER_PATH=/app/checkpoints/sudy-tokenizer
ENV PORT=8000
ENV HOST=0.0.0.0

CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
