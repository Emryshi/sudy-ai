import os
import torch
from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn
import json
import asyncio
import subprocess
import threading
import sys
import shutil
import re
from typing import Optional

from src.model import SudyConfig, SudyLMHeadModel, SudyTokenizer
from src.inference.search_utils import WebSearcher, ContextCompressor
from src.inference.safety import SafetyGuardrail

app = FastAPI(
    title="Sudy Inference & Control API",
    description="Gelişmiş MoE ve MLA Mimari Yapısına Sahip Türkçe LLM Çıkarım ve Yönetim Sunucusu",
    version="0.2.0"
)

# Load configuration and models
MODEL_PATH = os.getenv("MODEL_PATH", "./checkpoints/sudy-sft")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "./checkpoints/sudy-tokenizer")

model = None
tokenizer = None
device = None

searcher = WebSearcher()
compressor = ContextCompressor()
guardrail = SafetyGuardrail()

# Background Process Tracking Variables
active_process = None
active_task = None
process_logs = []
log_lock = threading.Lock()

@app.on_event("startup")
def startup_event():
    global model, tokenizer, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting API. Using device: {device}")

    # Load tokenizer
    if os.path.exists(TOKENIZER_PATH):
        print(f"Loading tokenizer from {TOKENIZER_PATH}")
        tokenizer = SudyTokenizer(TOKENIZER_PATH)
    else:
        print("Warning: Tokenizer path not found, initializing default tokenizer.")
        tokenizer = SudyTokenizer()

    # Load model configuration
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        config = SudyConfig.from_pretrained(MODEL_PATH)
        config.vocab_size = tokenizer.get_vocab_size()
        model = SudyLMHeadModel(config)
        
        # Load weights
        weights_file = os.path.join(MODEL_PATH, "model.pt")
        if os.path.exists(weights_file):
            model.load_state_dict(torch.load(weights_file, map_location="cpu"))
        model.to(device)
        model.eval()
    else:
        print("Warning: Model path not found. Initializing a tiny dummy model for API routing validation.")
        config = SudyConfig(
            vocab_size=tokenizer.get_vocab_size(),
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            d_latent=64,
            num_experts=4,
            top_k=1,
            num_mtp_heads=0
        )
        model = SudyLMHeadModel(config)
        model.to(device)
        model.eval()


# ---------------------------------------------------------
# Web UI Router
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def read_index():
    """Serves the custom single-page control panel HTML."""
    index_path = "src/inference/static/index.html"
    if not os.path.exists(index_path):
        return HTMLResponse(content="<h2>Static Index File Not Found</h2>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)


# ---------------------------------------------------------
# Generation Schemas & Endpoints
# ---------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Model için giriş metni/istem.")
    max_new_tokens: int = Field(50, ge=1, le=512, description="Üretilecek maksimum yeni token sayısı.")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Çeşitlilik sıcaklığı. 0.0 değeri tamamen kararlıdır.")
    top_k: int = Field(40, ge=0, le=100, description="Kısıtlanacak en yüksek olasılıklı kelime aday sayısı.")
    stream: bool = Field(False, description="Yanıtın akış (streaming) olarak verilip verilmeyeceği.")
    use_speculative: bool = Field(True, description="MTP tabanlı spekülatif çıkarım modunun kullanılıp kullanılmayacağı.")
    mtp_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Spekülatif token kabul eşiği.")
    web_search: bool = Field(False, description="Arama motoru üzerinden bilgi çekilip modele bağlam olarak verilip verilmeyeceği.")

class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    tokens_generated: int

async def token_generator(prompt: str, max_new_tokens: int, temperature: float, top_k: int, use_speculative: bool = True, mtp_threshold: float = 0.7):
    """Generates and streams tokens one by one, with speculative decoding support."""
    is_safe, msg = guardrail.check_prompt(prompt)
    if not is_safe:
        yield f"data: {json.dumps({'error': msg})}\n\n"
        yield "data: [DONE]\n\n"
        return
        
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    has_mtp = hasattr(model, "mtp_module") and model.config.num_mtp_heads > 0
    past_key_values = None
    num_tokens_to_feed = 1
    
    tokens_generated = 0
    with torch.no_grad():
        while tokens_generated < max_new_tokens:
            if past_key_values is not None:
                model_inputs = input_ids[:, -num_tokens_to_feed:]
            else:
                model_inputs = input_ids
                
            hidden_states, _, next_kv = model.model(
                model_inputs,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = next_kv
            
            logits_standard = model.lm_head(hidden_states[:, -1, :])
            if temperature == 0.0:
                next_token_1 = torch.argmax(logits_standard, dim=-1, keepdim=True)
            else:
                logits_1 = logits_standard / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits_1, min(top_k, logits_1.size(-1)))
                    logits_1[logits_1 < v[:, [-1]]] = -float("Inf")
                probs_1 = torch.softmax(logits_1, dim=-1)
                next_token_1 = torch.multinomial(probs_1, num_samples=1)
                
            token_id_1 = next_token_1.item()
            input_ids = torch.cat([input_ids, next_token_1], dim=-1)
            tokens_generated += 1
            num_tokens_to_feed = 1
            
            if token_id_1 == tokenizer.eos_token_id:
                word_1 = tokenizer.decode([token_id_1], skip_special_tokens=True)
                if word_1:
                    yield f"data: {json.dumps({'token': word_1})}\n\n"
                break
                
            word_1 = tokenizer.decode([token_id_1], skip_special_tokens=True)
            yield f"data: {json.dumps({'token': word_1})}\n\n"
            await asyncio.sleep(0.005)
            
            if tokens_generated >= max_new_tokens:
                break
                
            # Speculative token 2 using MTP
            if use_speculative and has_mtp and input_ids.shape[0] == 1:
                emb_next = model.model.embed_tokens(next_token_1)
                mtp_logits_list = model.mtp_module(hidden_states[:, -1:, :], [emb_next])
                logits_mtp = mtp_logits_list[0][:, -1, :]
                
                probs_mtp = torch.softmax(logits_mtp, dim=-1)
                max_prob, next_token_2 = torch.max(probs_mtp, dim=-1)
                
                if max_prob.item() > mtp_threshold:
                    token_id_2 = next_token_2.item()
                    next_token_2 = next_token_2.view(1, 1)
                    input_ids = torch.cat([input_ids, next_token_2], dim=-1)
                    tokens_generated += 1
                    num_tokens_to_feed = 2
                    
                    word_2 = tokenizer.decode([token_id_2], skip_special_tokens=True)
                    yield f"data: {json.dumps({'token': word_2, 'speculative': True})}\n\n"
                    await asyncio.sleep(0.005)
                    
                    if token_id_2 == tokenizer.eos_token_id:
                        break
                        
    yield "data: [DONE]\n\n"


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model veya Tokenizer yüklenemedi.")

    if len(request.prompt) > 2048:
        raise HTTPException(status_code=400, detail="Girdi istemi (prompt) en fazla 2048 karakter olmalıdır.")

    is_safe, msg = guardrail.check_prompt(request.prompt)
    if not is_safe:
        raise HTTPException(status_code=400, detail=msg)

    prompt = request.prompt
    if request.web_search:
        search_results = searcher.search(request.prompt, max_results=2)
        if search_results:
            documents = []
            for res in search_results:
                url = res["url"]
                snippet = res["snippet"]
                page_text = searcher.fetch_page_text(url)
                if page_text:
                    documents.append(page_text)
                else:
                    documents.append(snippet)
            
            compressed_context = compressor.compress(request.prompt, documents, max_tokens_approx=300)
            prompt = f"İnternet Arama Sonuçları:\n{compressed_context}\n\nSoru: {request.prompt}\nCevap:"

    if request.stream:
        return StreamingResponse(
            token_generator(
                prompt, 
                request.max_new_tokens, 
                request.temperature, 
                request.top_k,
                request.use_speculative,
                request.mtp_threshold
            ), 
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    
    # Non-streaming batch generation
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    if request.use_speculative and hasattr(model, "mtp_module") and model.config.num_mtp_heads > 0:
        generated_ids = model.generate_speculative(
            input_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            eos_token_id=tokenizer.eos_token_id,
            mtp_threshold=request.mtp_threshold
        )
    else:
        generated_ids = model.generate(
            input_ids, 
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            eos_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    prompt_decoded = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    
    if full_text.startswith(prompt_decoded):
        generated_text = full_text[len(prompt_decoded):].strip()
    else:
        generated_text = full_text
        
    is_safe_out, out_msg = guardrail.check_output(generated_text)
    if not is_safe_out:
        generated_text = out_msg
        
    return GenerateResponse(
        prompt=request.prompt,
        generated_text=generated_text,
        tokens_generated=len(generated_ids[0]) - len(prompt_ids)
    )


# ---------------------------------------------------------
# Dataset Management APIs
# ---------------------------------------------------------
def secure_filename(filename: str) -> str:
    filename = os.path.basename(filename)
    filename = re.sub(r'[^a-zA-Z0-9_\.\-]', '', filename)
    if not filename:
        filename = "uploaded_file.dat"
    return filename

@app.get("/api/datasets")
def list_datasets():
    """Lists files in data/raw and data/processed directories."""
    datasets = []
    for category in ["raw", "processed"]:
        dir_path = os.path.join("data", category)
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path):
                    datasets.append({
                        "name": file,
                        "category": category,
                        "size": os.path.getsize(file_path),
                        "path": f"data/{category}/{file}"
                    })
    return datasets

@app.get("/api/datasets/view")
def view_dataset(path: str = Query(..., description="Relative file path under data/")):
    """Returns the first 100 lines/JSON content of the dataset."""
    safe_path = os.path.normpath(path)
    normalized_path = safe_path.replace(os.path.sep, "/")
    if not (normalized_path.startswith("data/raw") or normalized_path.startswith("data/processed")):
        raise HTTPException(status_code=400, detail="Geçersiz dosya yolu.")
    
    if not os.path.exists(safe_path):
        raise HTTPException(status_code=404, detail="Dosya bulunamadı.")
        
    try:
        with open(safe_path, "r", encoding="utf-8", errors="ignore") as f:
            if safe_path.endswith(".json"):
                # Load JSON and return formatted
                data = json.load(f)
                # Slice list if it is a list of items to keep response small
                if isinstance(data, list):
                    data = data[:100]
                content = json.dumps(data, ensure_ascii=False, indent=2)
            else:
                lines = [f.readline() for _ in range(100)]
                content = "".join(lines)
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dosya okunamadı: {e}")

@app.delete("/api/datasets")
def delete_dataset(path: str = Query(..., description="Relative file path under data/")):
    """Deletes a dataset file."""
    safe_path = os.path.normpath(path)
    normalized_path = safe_path.replace(os.path.sep, "/")
    if not (normalized_path.startswith("data/raw") or normalized_path.startswith("data/processed")):
        raise HTTPException(status_code=400, detail="Geçersiz dosya silme talebi.")
        
    if os.path.exists(safe_path):
        os.remove(safe_path)
        return {"status": "success", "message": f"{path} başarıyla silindi."}
    else:
        raise HTTPException(status_code=404, detail="Dosya bulunamadı.")

@app.post("/api/datasets/upload")
async def upload_dataset(category: str = Query("raw", enum=["raw", "processed"]), file: UploadFile = File(...)):
    """Uploads a new dataset file and saves it in raw or processed directory."""
    filename = secure_filename(file.filename)
    target_dir = os.path.join("data", category)
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, filename)
    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Optional: If CSV uploaded to processed SFT, parse it directly to SFT JSON format
        if category == "processed" and filename.endswith(".csv"):
            import csv
            sft_data = []
            with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if headers and len(headers) >= 2:
                    for row in reader:
                        if len(row) >= 2:
                            sft_data.append({
                                "prompt": f"Aşağıdaki yorumun duygu analizini yapınız:\n\"{row[0]}\"\n\nAnaliz:",
                                "response": row[1]
                            })
            # Overwrite CSV with JSON format
            json_target = target_path.rsplit(".", 1)[0] + "_sft.json"
            with open(json_target, "w", encoding="utf-8") as json_f:
                json.dump(sft_data, json_f, ensure_ascii=False, indent=4)
            # Remove original raw CSV
            os.remove(target_path)
            return {"status": "success", "message": f"CSV başarıyla SFT JSON formatına dönüştürüldü ve kaydedildi: {json_target}"}

        return {"status": "success", "message": f"Dosya başarıyla yüklendi: {category}/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dosya kaydedilemedi: {e}")


# ---------------------------------------------------------
# Asynchronous Process Execution Management
# ---------------------------------------------------------
class TrainRequest(BaseModel):
    task: str  # "setup", "pretrain", "sft", "rlhf", "crawltrain"
    epochs: int = 2
    batch_size: int = 2
    lr: float = 3e-4
    vocab_size: int = 1000
    group_size: int = 4
    urls: Optional[str] = None
    data_path: Optional[str] = None

class HFDownloadRequest(BaseModel):
    repo_id: str
    split: str = "train"
    type: str = "sft" # sft or pretrain
    limit: int = 5000

def read_process_stdout(proc):
    global process_logs, active_process, active_task
    # Stream lines from output
    for line in iter(proc.stdout.readline, ""):
        with log_lock:
            process_logs.append(line)
    proc.stdout.close()
    proc.wait()
    with log_lock:
        process_logs.append(f"\n[PROCESS_COMPLETED with exit code {proc.returncode}]\n")
    active_process = None
    active_task = None

def start_background_process(cmd, task_name):
    global active_process, active_task, process_logs
    if active_process is not None:
        raise HTTPException(status_code=400, detail="Zaten çalışan aktif bir işlem var.")
    
    with log_lock:
        process_logs.clear()
        process_logs.append(f"[STARTING] {task_name} işlemi başlatılıyor...\n")
        
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        active_process = proc
        active_task = task_name
        
        t = threading.Thread(target=read_process_stdout, args=(proc,))
        t.daemon = True
        t.start()
        return {"status": "started", "task": task_name}
    except Exception as e:
        with log_lock:
            process_logs.append(f"[ERROR] İşlem başlatılamadı: {e}\n")
        raise HTTPException(status_code=500, detail=f"İşlem başlatılamadı: {e}")

@app.post("/api/train")
def run_training_task(req: TrainRequest):
    """Triggers background training tasks asynchronously."""
    if req.task == "setup":
        cmd = [sys.executable, "manage.py", "setup", "--vocab_size", str(req.vocab_size)]
    elif req.task == "pretrain":
        cmd = [
            sys.executable, "manage.py", "pretrain",
            "--epochs", str(req.epochs),
            "--batch_size", str(req.batch_size),
            "--lr", str(req.lr)
        ]
    elif req.task == "sft":
        data_path = req.data_path if req.data_path else "data/processed/sudy_sft_advanced.json"
        cmd = [
            sys.executable, "manage.py", "sft",
            "--epochs", str(req.epochs),
            "--batch_size", str(req.batch_size),
            "--lr", str(req.lr),
            "--data_path", data_path
        ]
    elif req.task == "rlhf":
        cmd = [
            sys.executable, "manage.py", "rlhf",
            "--epochs", str(req.epochs),
            "--batch_size", str(req.batch_size),
            "--lr", str(req.lr),
            "--group_size", str(req.group_size)
        ]
    elif req.task == "crawltrain":
        if not req.urls:
            raise HTTPException(status_code=400, detail="Lütfen taranacak web sitesi linklerini girin.")
        
        base_ckpt = "checkpoints/sudy-pretrain/model.pt"
        if os.path.exists("checkpoints/sudy-sft/model.pt"):
            base_ckpt = "checkpoints/sudy-sft/model.pt"
            
        cmd = [
            sys.executable, "scripts/crawl_and_train.py",
            "--urls", req.urls,
            "--config", "configs/sft.yaml",
            "--tokenizer_path", "checkpoints/sudy-tokenizer",
            "--sft_checkpoint", base_ckpt,
            "--output_dir", "checkpoints/sudy-sft",
            "--epochs", str(req.epochs),
            "--lr", str(req.lr)
        ]
    elif req.task == "autocrawl":
        if not req.urls:
            raise HTTPException(status_code=400, detail="Lütfen aranacak web terimlerini girin.")
        cmd = [
            sys.executable, "scripts/autonomous_crawler.py",
            "--query", req.urls,
            "--max_results", "5"
        ]
    else:
        raise HTTPException(status_code=400, detail="Geçersiz eğitim görevi.")
        
    return start_background_process(cmd, f"Training: {req.task}")

@app.post("/api/datasets/download_hf")
def download_hf_dataset_endpoint(req: HFDownloadRequest):
    """Triggers background HuggingFace dataset download asynchronously."""
    cmd = [
        sys.executable, "scripts/download_hf_dataset.py",
        "--repo_id", req.repo_id,
        "--split", req.split,
        "--type", req.type,
        "--limit", str(req.limit)
    ]
    return start_background_process(cmd, f"Download HF: {req.repo_id}")

@app.post("/api/train/stop")
def stop_training_task():
    """Kills the active training/testing process if running."""
    global active_process
    if active_process is None:
        return {"status": "no process running"}
    try:
        active_process.terminate()
        active_process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        active_process.kill()
    return {"status": "stopped"}

@app.get("/api/train/status")
def get_train_status():
    """Returns active task name and recent logs count."""
    global active_process, active_task, process_logs
    return {
        "running": active_process is not None,
        "task": active_task,
        "log_count": len(process_logs)
    }

@app.get("/api/train/logs")
def get_train_logs(last_idx: int = 0):
    """Returns new log lines since last_idx via lightweight JSON polling."""
    global process_logs, active_process, active_task
    with log_lock:
        logs_slice = process_logs[last_idx:]
        next_idx = len(process_logs)
    return {
        "logs": logs_slice,
        "next_idx": next_idx,
        "running": active_process is not None,
        "task": active_task
    }

@app.get("/api/train/stream_logs")
async def stream_logs():
    """Streams active process output in real-time using Server-Sent Events (SSE)."""
    async def log_generator():
        global process_logs, active_process
        idx = 0
        while True:
            # Yield accumulated logs
            if idx < len(process_logs):
                yield f"data: {process_logs[idx]}\n\n"
                # If completion string reached, terminate stream
                if "[PROCESS_COMPLETED" in process_logs[idx]:
                    break
                idx += 1
            else:
                # If no active process and we printed all logs, end
                if active_process is None:
                    # Double check if any log appended right before thread finished
                    if idx >= len(process_logs):
                        break
                await asyncio.sleep(0.1)
                
    return StreamingResponse(log_generator(), media_type="text/event-stream")

@app.post("/api/test")
def run_unit_tests():
    """Asynchronously starts unit testing suite."""
    cmd = [sys.executable, "manage.py", "test"]
    return start_background_process(cmd, "Testing: pytest")


# ---------------------------------------------------------
# FastAPI Health Check
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "healthy", "device": str(device)}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("src.inference.api:app", host=host, port=port, reload=True)
