from fastapi import FastAPI, Request, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
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
import re

MODEL_DIRS = {
    'gold': 'model/gold',
    'silver': 'model/silver',
    'platinum': 'model/platinum',
    'palladium': 'model/palladium',
}

loaded_models = {}
loaded_tokenizers = {}


def load_model_and_tokenizer(metal):
    if metal in loaded_models and metal in loaded_tokenizers:
        return loaded_models[metal], loaded_tokenizers[metal]
    model_dir = MODEL_DIRS[metal]
    with open(os.path.join(model_dir, 'adapter_config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
    base_model_name = config['base_model_name_or_path']
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
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


class PredictionRequest(BaseModel):
    date: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    news: list[str]


class PredictionResponse(BaseModel):
    label: str
    justification: str


def format_prompt_enhanced(input_text, tokenizer):
    example_1_input = "Дата: 2023-01-10. Золото OHLCV: Open=1750.0, High=1755.0, Low=1748.0, Close=1752.0, Volume=800. Новости дня: Инфляция снижается / Курс доллара укрепляется / Рынок акций стабилен."
    example_1_output = "Метка: Sell\\nОбоснование: Укрепление доллара и снижение инфляции обычно негативно сказываются на стоимости золота, поскольку оно становится менее привлекательным в качестве средства сохранения стоимости. Стабильный рынок акций также снижает спрос на безопасные активы."

    example_2_input = "Дата: 2023-01-11. Золото OHLCV: Open=1752.0, High=1760.0, Low=1751.0, Close=1758.0, Volume=1100. Новости дня: Геополитическая напряженность растет на Ближнем Востоке / Центральные банки рассматривают смягчение монетарной политики."
    example_2_output = "Метка: Buy\\nОбоснование: Рост геополитической напряженности увеличивает спрос на золото как безопасный актив. Возможное смягчение монетарной политики центральных банков также способствует росту цен на золото из-за ожиданий увеличения ликвидности и инфляции."

    example_3_input = "Дата: 2023-01-12. Золото OHLCV: Open=1758.0, High=1759.0, Low=1757.0, Close=1758.0, Volume=900. Новости дня: Экономические данные смешанные / Нет значительных изменений на мировых рынках / Торги спокойные."
    example_3_output = "Метка: Hold\\nОбоснование: Отсутствие явных экономических или геополитических триггеров и стабильные показатели OHLCV указывают на отсутствие причин для изменения текущей позиции. Рынок находится в состоянии неопределенности."

    system_prompt = (
        "Ты опытный финансовый аналитик, специализирующийся на рынке золота. Твоя задача — проанализировать "
        "предоставленные данные по цене золота (OHLCV) и новостям дня, а затем принять обоснованное инвестиционное решение: "
        "Buy (Покупка), Hold (Удержание) или Sell (Продажа).\\n\\n"
        "Твой анализ должен быть глубоким и содержать:\\n"
        "1.  **Оценку ценовых движений и объема:** Как текущие цены (Open, High, Low, Close) и объем торгов (Volume) соотносятся "
        "с общим трендом или диапазоном? Есть ли признаки прорыва, консолидации или разворота? Не уделяй внимания объему.\\n"
        "2.  **Анализ релевантности новостей:** Какие из новостей дня напрямую или косвенно влияют на цену золота? Например, новости "
        "о монетарной политике, инфляции, геополитической нестабильности, макроэкономических показателях США или крупнейших экономик могут быть "
        "важны для золота как защитного актива. Игнорируй новости, не имеющие отношения к рынку золота или общие корпоративные новости, "
        "если они не влияют на общие экономические настроения. Обращай внимания на молейшие сигналы для изменения цены на момент следующей торговой недели\\n"
        "3.  **Обоснование решения:** Четко объясни, почему было выбрано именно это решение (Buy, Hold, Sell), ссылаясь на конкретные "
        "технические или фундаментальные факторы из предоставленных данных. Твое обоснование должно быть кратким, но информативным (2-4 предложения).\\n\\n"
        "Если данных недостаточно для принятия четкого решения, или если рынок находится в состоянии неопределенности, обоснуй, "
        "почему 'Hold' является наиболее разумной стратегией, и чего ты ожидаешь для изменения этого решения. Но старайся не злоупотреблять HOLD, если есть возможность, лучше использовать другие метки.\\n\\n"
        "Представь ответ в СТРОГОМ формате:\\nМетка: [Hold/Buy/Sell]\\nОбоснование: [Твоё подробное объяснение]"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Данные:\\n{example_1_input}\\n{example_1_output}"},
        {"role": "user", "content": f"Данные:\\n{example_2_input}\\n{example_2_output}"},
        {"role": "user", "content": f"Данные:\\n{example_3_input}\\n{example_3_output}"},
        {"role": "user", "content": f"Данные:\\n{input_text}\\n"}
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_label_and_justification_improved(generated_text):
    original = generated_text
    label = "UNKNOWN"
    justification = "Не удалось извлечь обоснование."

    generated_text = re.sub(r"^(.*?)(Метка:|Рекомендация:|Buy|Sell|Hold)", r"\\2", generated_text,
                            flags=re.DOTALL | re.IGNORECASE)

    label_pattern = r"(?:Метка:|Рекомендация:|Decision:|Label:)\\s*(buy|sell|hold)"
    label_match = re.search(label_pattern, generated_text, re.IGNORECASE)

    if label_match:
        label = label_match.group(1).upper()
        start_idx = label_match.end()
    else:
        for keyword in ["Buy", "Sell", "Hold"]:
            if keyword.lower() in generated_text.lower():
                label = keyword.upper()
                start_idx = generated_text.lower().find(keyword.lower()) + len(keyword)
                break
        else:
            for keyword in ["Buy", "Sell", "Hold"]:
                if keyword.lower() in generated_text[-100:].lower():
                    label = keyword.upper()
                    break

    if label != "UNKNOWN":
        after_label = generated_text[start_idx:]
        justification_end = re.search(r"(Buy|Sell|Hold|---|\\[|$)", after_label, re.IGNORECASE)
        if justification_end:
            justification = after_label[:justification_end.start()].strip()
        else:
            justification = after_label.strip()
    else:
        justification = generated_text.strip()

    justification = re.sub(r"^(Обоснование:|Justification:)\\s*", "", justification, flags=re.IGNORECASE).strip()
    return label, justification


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
            if period == 'year':
                check_query = f"SELECT MIN(timestamp) as min_date FROM {table_name};"
                cur.execute(check_query)
                result = cur.fetchone()
                if result and result['min_date']:
                    min_db_date = result['min_date']
                    one_year_ago = datetime.now() - timedelta(days=365)
                    if min_db_date > one_year_ago:
                        query = f"""
                            SELECT timestamp, "Close" as price
                            FROM {table_name}
                            ORDER BY timestamp;
                        """
                        cur.execute(query)
                    else:
                        cur.execute(query, (start_date, end_date))
                else:
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


@app.post("/predict/{metal}", response_model=PredictionResponse)
async def predict(metal: str, request: PredictionRequest):
    if metal.lower() not in MODEL_DIRS:
        return JSONResponse(status_code=404, content={"error": "Metal not found"})

    try:
        model, tokenizer = load_model_and_tokenizer(metal.lower())
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to load model: {str(e)}"})

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

    news_str = " / ".join(request.news)
    input_text = (
        f"Дата: {request.date}. Золото OHLCV: Open={request.open_price}, High={request.high_price}, "
        f"Low={request.low_price}, Close={request.close_price}, Volume={request.volume}. "
        f"Новости дня: {news_str}"
    )

    prompt = format_prompt_enhanced(input_text, tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=model.config.max_position_embeddings).to(device)

    generation_config = GenerationConfig(
        max_new_tokens=400,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        repetition_penalty=1.1
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=generation_config,
        )

    generated_tokens_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    predicted_text = tokenizer.decode(generated_tokens_ids, skip_special_tokens=True).strip()

    predicted_label, justification = extract_label_and_justification_improved(predicted_text)

    return PredictionResponse(label=predicted_label, justification=justification)