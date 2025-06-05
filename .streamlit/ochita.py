import streamlit as st
st.set_page_config(page_title="Metals Dashboard", layout="wide")
from streamlit_extras.metric_cards import style_metric_cards
from csv_scraper import load_all_csv, get_period_df
import requests
import os
import json
import torch

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 0; width: 100%; }
    .stTabs [data-baseweb="tab"]      { height:10px; width:100%; padding:15px 0; font-size:32px; }
    .stTabs [data-baseweb="tab-panel"]{ padding-top:0; }
    .stMetric, .stMarkdown            { text-align:center; }
    .stMetric [data-testid="stMetricLabel"]{ font-size:54px; font-weight:bold; }
    .stMetric [data-testid="stMetricValue"]{ font-size:40px; font-weight:bold; }
    .stMetric [data-testid="stMetricDelta"]{ font-size:25px; transform:scaleX(.97) translateX(-3.5%); }
    .stMetric, .stMetric > div {
        background-color: #021024 !important;   /* любой тёмный оттенок */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────── Данные ───────────────────
all_data = load_all_csv("../aggregator/dataset/datasets/labeled_dataset")  

# ─────────────────── Логика показа блока ───────────────────
def show_info(metal: str, period_tab, period_key: str):
    df = all_data[metal]
    series = get_period_df(df, period_key)       # price vs time

    # Берём крайние точки выбранного диапазона
    first_val = series.iloc[0, 0]
    last_val  = series.iloc[-1, 0]
    delta     = last_val - first_val
    pct       = (delta / first_val * 100) if first_val else 0

    col_price, col_pct = period_tab.columns(2, vertical_alignment="center")
    col_price.metric("Today price",      f"{last_val:,.2f} $/oz", f"{delta:+.2f}")
    col_pct.metric("Percentage growth", f"{pct:.2f} %",     f"{pct:+.2f} %")
    style_metric_cards()

    period_tab.line_chart(series)

# ─────────────────── UI как раньше ───────────────────
metals_container = st.container()
gold_tab, silver_tab, platinum_tab, palladium_tab = metals_container.tabs(
    ["Gold", "Silver", "Platinum", "Palladium"]
)

# ─────────────────── Ссылки ───────────────────
st.markdown("# Buy metal in banks")
sber, tinkoff, vtb, rsh = st.columns(4)
sber.link_button("Sber",     "https://www.sberbank.com/ru/person/metall", use_container_width=True)
tinkoff.link_button("T-Bank", "https://www.tbank.ru/invest/promo/gold/?ysclid=m9mn69syar893212723", use_container_width=True)
vtb.link_button("VTB",       "https://www.vtb.ru/personal/vklady-i-scheta/obezlichennyj-metallicheskij-schet/", use_container_width=True)
rsh.link_button("Rosselhoz", "https://www.rshb.ru/natural/investments/metal-gold", use_container_width=True)

# ─────────────────── Логика добавления новых функций ───────────────────
FASTAPI_URL = "http://localhost:8000/batch_predict"

# Helper to get batch predictions from FastAPI
@st.cache_data(show_spinner=False)
def get_batch_predictions(metal, jsonl_path, num_news):
    with open(jsonl_path, "rb") as f:
        files = {"file": (os.path.basename(jsonl_path), f, "application/jsonl")}
        response = requests.post(FASTAPI_URL, params={"metal": metal, "max_samples": num_news}, files=files, timeout=1200)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}

# ─────────────────── GOLD ───────────────────
with gold_tab:
    st.container().markdown("# Here is some info about gold")
    week_tab, month_tab, year_tab = gold_tab.tabs(["Week", "Month", "Year"])
    show_info("gold", week_tab,  "week")
    show_info("gold", month_tab, "month")
    show_info("gold", year_tab,  "year")
    additional_info_container = st.container(border=True)
    with additional_info_container:
        st.markdown("### Our predictions:")
        jsonl_path = "../aggregator/dataset/datasets/labeled_dataset/json_data_gold.jsonl"
        num_news = st.number_input("Number of latest news to predict", min_value=1, max_value=100, value=20, key="gold_num_news")
        if st.button("Predict", key="predict_gold"):
            with st.spinner("Predicting for gold..."):
                result = get_batch_predictions("gold", jsonl_path, num_news)
            if "error" in result:
                st.error(result["error"])
            else:
                st.write(f"Accuracy: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
                st.dataframe([{k: v for k, v in row.items() if k in ["input_text", "true_label", "prediction"]} for row in result["results"]], use_container_width=True)

# ─────────────────── SILVER ───────────────────
with silver_tab:
    st.container().markdown("# Here is some info about silver")
    week_tab, month_tab, year_tab = silver_tab.tabs(["Week", "Month", "Year"])
    show_info("silver", week_tab,  "week")
    show_info("silver", month_tab, "month")
    show_info("silver", year_tab,  "year")
    additional_info_container = st.container(border=True)
    with additional_info_container:
        st.markdown("### Our predictions:")
        jsonl_path = "../aggregator/dataset/datasets/labeled_dataset/json_data_silver.jsonl"
        num_news = st.number_input("Number of latest news to predict", min_value=1, max_value=100, value=20, key="silver_num_news")
        if st.button("Predict", key="predict_silver"):
            with st.spinner("Predicting for silver..."):
                result = get_batch_predictions("silver", jsonl_path, num_news)
            if "error" in result:
                st.error(result["error"])
            else:
                st.write(f"Accuracy: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
                st.dataframe([{k: v for k, v in row.items() if k in ["input_text", "true_label", "prediction"]} for row in result["results"]], use_container_width=True)

# ─────────────────── PLATINUM ───────────────────
with platinum_tab:
    st.container().markdown("# Here is some info about platinum")
    week_tab, month_tab, year_tab = platinum_tab.tabs(["Week", "Month", "Year"])
    show_info("platinum", week_tab,  "week")
    show_info("platinum", month_tab, "month")
    show_info("platinum", year_tab,  "year")
    additional_info_container = st.container(border=True)
    with additional_info_container:
        st.markdown("### Our predictions:")
        jsonl_path = "../aggregator/dataset/datasets/labeled_dataset/json_data_platinum.jsonl"
        num_news = st.number_input("Number of latest news to predict", min_value=1, max_value=100, value=20, key="platinum_num_news")
        if st.button("Predict", key="predict_platinum"):
            with st.spinner("Predicting for platinum..."):
                result = get_batch_predictions("platinum", jsonl_path, num_news)
            if "error" in result:
                st.error(result["error"])
            else:
                st.write(f"Accuracy: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
                st.dataframe([{k: v for k, v in row.items() if k in ["input_text", "true_label", "prediction"]} for row in result["results"]], use_container_width=True)

# ─────────────────── PALLADIUM ───────────────────
with palladium_tab:
    st.container().markdown("# Here is some info about palladium")
    week_tab, month_tab, year_tab = palladium_tab.tabs(["Week", "Month", "Year"])
    show_info("palladium", week_tab,  "week")
    show_info("palladium", month_tab, "month")
    show_info("palladium", year_tab,  "year")
    additional_info_container = st.container(border=True)
    with additional_info_container:
        st.markdown("### Our predictions:")
        jsonl_path = "../aggregator/dataset/datasets/labeled_dataset/json_data_palladium.jsonl"
        num_news = st.number_input("Number of latest news to predict", min_value=1, max_value=100, value=20, key="palladium_num_news")
        if st.button("Predict", key="predict_palladium"):
            with st.spinner("Predicting for palladium..."):
                result = get_batch_predictions("palladium", jsonl_path, num_news)
            if "error" in result:
                st.error(result["error"])
            else:
                st.write(f"Accuracy: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
                st.dataframe([{k: v for k, v in row.items() if k in ["input_text", "true_label", "prediction"]} for row in result["results"]], use_container_width=True)
