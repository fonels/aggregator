import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
from csv_scraper import load_all_csv, get_period_df

st.set_page_config(page_title="Metals Dashboard", layout="wide")
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
        background-color: #161616 !important;   /* любой тёмный оттенок */
        border: 1px solid #333 !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────── Данные ───────────────────
all_data = load_all_csv("../dataset/datasets/labeled_dataset")  

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
    col_price.metric("Today price",      f"{last_val:,.2f}", f"{delta:+.2f}")
    col_pct.metric("Percentage growth", f"{pct:.2f} %",     f"{pct:+.2f} %")
    style_metric_cards()

    period_tab.line_chart(series)

# ─────────────────── UI как раньше ───────────────────
metals_container = st.container()
gold_tab, silver_tab, platinum_tab, palladium_tab = metals_container.tabs(
    ["Gold", "Silver", "Platinum", "Palladium"]
)

# ——— GOLD ———
with gold_tab:
    st.container().markdown("# Here is some info about gold")
    day_tab, week_tab, month_tab, year_tab = gold_tab.tabs(["Day", "Week", "Month", "Year"])
    show_info("gold", day_tab,   "day")
    show_info("gold", week_tab,  "week")
    show_info("gold", month_tab, "month")
    show_info("gold", year_tab,  "year")
    st.container(border=True).write("blah blah blah blah")

# ——— SILVER ———
with silver_tab:
    st.container().markdown("# Here is some info about silver")
    day_tab, week_tab, month_tab, year_tab = silver_tab.tabs(["Day", "Week", "Month", "Year"])
    show_info("silver", day_tab,   "day")
    show_info("silver", week_tab,  "week")
    show_info("silver", month_tab, "month")
    show_info("silver", year_tab,  "year")
    st.container(border=True).write("blah blah blah blah")

# ——— PLATINUM ———
with platinum_tab:
    st.container().markdown("# Here is some info about platinum")
    day_tab, week_tab, month_tab, year_tab = platinum_tab.tabs(["Day", "Week", "Month", "Year"])
    show_info("platinum", day_tab,   "day")
    show_info("platinum", week_tab,  "week")
    show_info("platinum", month_tab, "month")
    show_info("platinum", year_tab,  "year")
    st.container(border=True).write("blah blah blah blah")

# ——— PALLADIUM ———
with palladium_tab:
    st.container().markdown("# Here is some info about palladium")
    day_tab, week_tab, month_tab, year_tab = palladium_tab.tabs(["Day", "Week", "Month", "Year"])
    show_info("palladium", day_tab,   "day")
    show_info("palladium", week_tab,  "week")
    show_info("palladium", month_tab, "month")
    show_info("palladium", year_tab,  "year")
    st.container(border=True).write("blah blah blah blah")

# ─────────────────── Ссылки ───────────────────
st.markdown("# Buy metal in banks")
sber, tinkoff, vtb, rsh = st.columns(4)
sber.link_button("Sber",     "https://www.sberbank.com/ru/person/metall", use_container_width=True)
tinkoff.link_button("T-Bank", "https://www.tbank.ru/invest/promo/gold/?ysclid=m9mn69syar893212723", use_container_width=True)
vtb.link_button("VTB",       "https://www.vtb.ru/personal/vklady-i-scheta/obezlichennyj-metallicheskij-schet/", use_container_width=True)
rsh.link_button("Rosselhoz", "https://www.rshb.ru/natural/investments/metal-gold", use_container_width=True)
