# Sudy Dağıtım Kılavuzu

Sudy modelinizi API servisi olarak yayına almak için FastAPI ve Docker yapılandırmaları hazır durumdadır.

## 1. FastAPI ile Yerel Başlatma
Bağımlılıkları kurduktan sonra uvicorn sunucusunu başlatabilirsiniz.

```bash
# Bağımlılıkları geliştirici modunda kurun
pip install -e .

# API sunucusunu başlatın
MODEL_PATH=./checkpoints/sudy-sft TOKENIZER_PATH=./checkpoints/sudy-tokenizer uvicorn src.inference.api:app --reload --host 0.0.0.0 --port 8000
```

### API İstek Örnekleri

#### Toplu Çıkarım (Batch Generate - POST `/generate`)
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "türkiye nin başkenti neresidir", "max_new_tokens": 20, "temperature": 0.0, "stream": false}'
```

#### Akış Çıkarımı (Streaming Generate - POST `/generate`)
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "yapay zeka nedir", "max_new_tokens": 30, "temperature": 0.7, "stream": true}'
```

---

## 2. Docker ve Docker Compose ile Dağıtım
Sunucu veya GPU barındıran ortamlarda Docker kullanarak izole bir servis ayağa kaldırabilirsiniz.

```bash
# Docker imajını derleyin ve Compose ile başlatın
docker-compose -f docker/docker-compose.yml up --build -d
```

API servisiniz `http://localhost:8000` adresinde yayına başlayacaktır. Güncellemeler ve modeller `./checkpoints` dizininden Docker konteynerine otomatik olarak yansıtılacaktır.
