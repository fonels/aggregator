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
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from data_pipeline.config import DB_CONFIG, TABLE_MAPPINGS
from datetime import datetime, timedelta

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

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
    finally:
        if conn:
            conn.close()

app = FastAPI()

@app.get("/data/{metal}")
def get_metal_data(metal: str, period: str = 'year'):
    table_name = f"{metal.lower()}_data"
    if table_name not in TABLE_MAPPINGS.values():
        return JSONResponse(status_code=404, content={"error": "Metal not found"})

    end_date = datetime.now()
    if period == 'week':
        start_date = end_date - timedelta(days=7)
    elif period == 'month':
        start_date = end_date - timedelta(days=30)
    elif period == 'year':
        start_date = end_date - timedelta(days=365)
    else:
        return JSONResponse(status_code=400, content={"error": "Invalid period specified"})

    query = f"""
        SELECT timestamp, "Close" as price
        FROM {table_name}
        WHERE timestamp >= %s AND timestamp <= %s
        ORDER BY timestamp;
    """
    
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # First, check if there is any data for the last year for 'year' period
            if period == 'year':
                check_query = f"SELECT MIN(timestamp) as min_date FROM {table_name};"
                cur.execute(check_query)
                result = cur.fetchone()
                if result and result['min_date']:
                    min_db_date = result['min_date']
                    one_year_ago = datetime.now() - timedelta(days=365)
                    if min_db_date > one_year_ago:
                        # if the earliest data is more recent than one year, fetch all data
                        query = f"""
                            SELECT timestamp, "Close" as price
                            FROM {table_name}
                            ORDER BY timestamp;
                        """
                        cur.execute(query)
                    else:
                        cur.execute(query, (start_date, end_date))
                else: # no data in table
                     cur.execute(query, (start_date, end_date))
            else:
                cur.execute(query, (start_date, end_date))

            data = cur.fetchall()

    return data

@app.get("/news")
def get_news():
    table_name = f"news_data"
    if table_name not in TABLE_MAPPINGS.values():
        return JSONResponse(status_code=404, content={"error": "News not found"})

    query = f"""
        SELECT "headlines" as title
        FROM news_headlines
        ORDER BY timestamp DESC LIMIT 1;
    """
    
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            data = cur.fetchall()
    
    news_titles_list = data[0]['title'].split(' / ')
    return news_titles_list

#здесь функция, которая отправляет запрос к модели
