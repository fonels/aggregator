# model_api.py
from fastapi import FastAPI, Request, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os
import json
import io

# Map metal to model directory
MODEL_DIRS = {
    'gold': 'model/gold',
    'silver': 'model/silver',
    'platinum': 'model/platinum',
    'palladium': 'model/palladium',
}

# Cache loaded models and tokenizers
loaded_models = {}
loaded_tokenizers = {}

def load_model_and_tokenizer(metal):
    if metal in loaded_models and metal in loaded_tokenizers:
        return loaded_models[metal], loaded_tokenizers[metal]
    model_dir = MODEL_DIRS[metal]
    # Load base model name from adapter_config.json
    with open(os.path.join(model_dir, 'adapter_config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
    base_model_name = config['base_model_name_or_path']
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
    # Load LoRA adapter (uses .safetensors if present)
    model = PeftModel.from_pretrained(base_model, model_dir, adapter_name=None)
    model.eval()
    loaded_models[metal] = model
    loaded_tokenizers[metal] = tokenizer
    return model, tokenizer

app = FastAPI()

class PredictRequest(BaseModel):
    metal: str  # 'gold', 'silver', 'platinum', 'palladium'
    title: str  # news title

@app.post('/predict')
async def predict(request: PredictRequest):
    metal = request.metal.lower()
    title = request.title
    if metal not in MODEL_DIRS:
        return {"error": f"Unknown metal: {metal}"}
    model, tokenizer = load_model_and_tokenizer(metal)
    # Format input as chat template
    chat_template = [
        {"role": "user", "content": title},
        {"role": "assistant", "content": ""}
    ]
    prompt = tokenizer.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=32)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract only the assistant's answer (after the user prompt)
    answer = decoded.split(title)[-1].strip()
    return {"prediction": answer}

# ──────────────── Batch prediction endpoint ────────────────
@app.post('/batch_predict')
async def batch_predict(
    metal: str, 
    file: UploadFile = File(...), 
    period: str = Query(None), 
    max_samples: int = Query(20)
):
    metal = metal.lower()
    if metal not in MODEL_DIRS:
        return JSONResponse(status_code=400, content={"error": f"Unknown metal: {metal}"})
    model, tokenizer = load_model_and_tokenizer(metal)
    # Read JSONL file
    contents = await file.read()
    lines = contents.decode('utf-8').splitlines()
    results = []
    correct = 0
    total = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            input_text = item.get('input_text', '')
            true_label = item.get('output_label', '')

            # Filter by period if specified
            if period and period.lower() not in input_text.lower():
                continue

            # Only use news titles (extract from input_text)
            # Example: extract after 'Новости дня:'
            if 'Новости дня:' in input_text:
                news_titles = input_text.split('Новости дня:')[-1].strip()
            else:
                news_titles = input_text

            # Format input as chat template
            chat_template = [
                {"role": "user", "content": news_titles},
                {"role": "assistant", "content": ""}
            ]
            prompt = tokenizer.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=32)
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            pred = decoded.split(news_titles)[-1].strip()
            results.append({
                "input_text": news_titles,
                "true_label": true_label,
                "prediction": pred
            })
            if pred.lower() == true_label.lower():
                correct += 1
            total += 1
            if total >= max_samples:
                break
        except Exception as e:
            results.append({"error": str(e), "line": line})
    accuracy = correct / total if total > 0 else 0.0
    return {"results": results, "accuracy": accuracy, "total": total, "correct": correct}
