import streamlit as st
st.set_page_config(page_title="Metals Dashboard", layout="wide")
from streamlit_extras.metric_cards import style_metric_cards
import requests
import os
import pandas as pd

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

# ─────────────────── Data ───────────────────
API_URL = "http://localhost:8000/data"
NEWS_URL = "http://localhost:8000/news"

@st.cache_data(ttl=600) # Cache data for 10 minutes
def get_data_from_server(metal: str, period: str) -> pd.DataFrame:
    """Fetches data from the FastAPI server and returns a DataFrame."""
    try:
        response = requests.get(f"{API_URL}/{metal}", params={"period": period}, timeout=30)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        df['price'] = pd.to_numeric(df['price'])
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Could not fetch data from server: {e}")
        return pd.DataFrame()

def get_news_from_server() -> pd.DataFrame:
    """Fetches news from the FastAPI server and returns a DataFrame."""
    try:
        response = requests.get(f"{NEWS_URL}", timeout=30)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Could not fetch news from server: {e}")
        return pd.DataFrame()

# ─────────────────── Logic for showing block ───────────────────
def show_info(metal: str, period_tab, period_key: str):
    series_df = get_data_from_server(metal, period_key)

    if series_df.empty:
        period_tab.warning("No data available for this period.")
        return

    # Take the extreme points of the selected range
    first_val = series_df['price'].iloc[0]
    last_val  = series_df['price'].iloc[-1]
    delta     = last_val - first_val
    pct       = (delta / first_val * 100) if first_val else 0

    col_price, col_pct = period_tab.columns(2, vertical_alignment="center")
    col_price.metric("Today price",      f"{last_val:,.2f} $/oz", f"{delta:+.2f}")
    col_pct.metric("Percentage growth", f"{pct:.2f} %",     f"{pct:+.2f} %")
    style_metric_cards()

    period_tab.line_chart(series_df.rename(columns={'price': 'Price'}))

# ─────────────────── UI as before ───────────────────
metals_container = st.container()
gold_tab, silver_tab, platinum_tab, palladium_tab = metals_container.tabs(
    ["Gold", "Silver", "Platinum", "Palladium"]
)

# ─────────────────── News ───────────────────
news_container = st.container(border=True)
news_container.markdown("# Here is some news")
news_titles_container = news_container.container(border=True)
news_titles = get_news_from_server()
for title in news_titles:
    news_titles_container.markdown(title)

# ─────────────────── Ссылки ───────────────────
st.markdown("# Buy metal in banks")
sber, tinkoff, vtb, rsh = st.columns(4)
sber.link_button("Sber",     "https://www.sberbank.com/ru/person/metall", use_container_width=True)
tinkoff.link_button("T-Bank", "https://www.tbank.ru/invest/promo/gold/?ysclid=m9mn69syar893212723", use_container_width=True)
vtb.link_button("VTB",       "https://www.vtb.ru/personal/vklady-i-scheta/obezlichennyj-metallicheskij-schet/", use_container_width=True)
rsh.link_button("Rosselhoz", "https://www.rshb.ru/natural/investments/metal-gold", use_container_width=True)

# ─────────────────── Логика добавления новых функций ───────────────────

# ─────────────────── GOLD ───────────────────
with gold_tab:
    st.container().markdown("# Here is some info about gold")
    week_tab, month_tab, year_tab = gold_tab.tabs(["Week", "Month", "Year"])
    show_info("gold", week_tab,  "week")
    show_info("gold", month_tab, "month")
    show_info("gold", year_tab,  "year")
    additional_info_container = st.container(border=True)
    with additional_info_container:
        st.markdown("### Our predictions for the next trading week:")
        if st.button("Make prediction", key="predict_gold"):
            with st.spinner("Getting prediction for gold..."):
                st.success("Prediction for gold is here")

# ─────────────────── SILVER ───────────────────
with silver_tab:
    st.container().markdown("# Here is some info about silver")
    week_tab, month_tab, year_tab = silver_tab.tabs(["Week", "Month", "Year"])
    show_info("silver", week_tab,  "week")
    show_info("silver", month_tab, "month")
    show_info("silver", year_tab,  "year")
    additional_info_container = st.container(border=True)
    with additional_info_container:
        st.markdown("### Our predictions for the next trading week:")
        if st.button("Make prediction", key="predict_silver"):
            with st.spinner("Getting prediction for silver..."):
                st.success("Prediction for silver is here")
# ─────────────────── PLATINUM ───────────────────
with platinum_tab:
    st.container().markdown("# Here is some info about platinum")
    week_tab, month_tab, year_tab = platinum_tab.tabs(["Week", "Month", "Year"])
    show_info("platinum", week_tab,  "week")
    show_info("platinum", month_tab, "month")
    show_info("platinum", year_tab,  "year")
    additional_info_container = st.container(border=True)
    with additional_info_container:
        st.markdown("### Our predictions for the next trading week:")
        if st.button("Make prediction", key="predict_platinum"):
            with st.spinner("Getting prediction for platinum..."):
                st.success("Prediction for platinum is here")

# ─────────────────── PALLADIUM ───────────────────
with palladium_tab:
    st.container().markdown("# Here is some info about palladium")
    week_tab, month_tab, year_tab = palladium_tab.tabs(["Week", "Month", "Year"])
    show_info("palladium", week_tab,  "week")
    show_info("palladium", month_tab, "month")
    show_info("palladium", year_tab,  "year")
    additional_info_container = st.container(border=True)
    with additional_info_container:
        st.markdown("### Our predictions for the next trading week:")
        if st.button("Make prediction", key="predict_palladium"):
            with st.spinner("Getting prediction for palladium..."):
                st.success("Prediction for palladium is here")


