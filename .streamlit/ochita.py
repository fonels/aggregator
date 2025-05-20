import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
import pandas as pd #это для примера графика
import numpy as np  #и это

# CSS для стилизации
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        height: 10px;
        width: 100%;
        padding: 15px 0;
        font-size: 32px;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0;
    }
    .stMetric {
        text-align: center;
    }
    .stMarkdown {
        text-align: center;
    }
    /* Metric container styling */
    .stMetric [data-testid="stMetricLabel"] {
        font-size: 54px;
        text-align: center;
        font-weight: bold;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        position: relative;
    }
    .stMetric [data-testid="stMetricDelta"] {
        font-size: 25px;
        position: relative;
        transform: scaleX(0.97) translateX(-3.5%)
    }
    </style>
""", unsafe_allow_html=True)

def show_info(metal, period):
    currency_col, percent_col = period.columns(2, vertical_alignment='center')

    metric_container_currency = currency_col.container(border=True) #контейнеры с инфой по цене металла
    metric_container_currency.metric(label='Today price', value=1, delta=-23) #тут надо взять информацию из датасета

    metric_container_percents = percent_col.container(border=True)
    metric_container_percents.metric(label='Percentage growth', value=10, delta=0.24) #тут тоже
            
    day_chart_container = period.container(border=True)
    day_chart_container.line_chart(pd.DataFrame(np.random.randn(20, 1), columns=["a"]))



#выбор металла для отображения информации
metals_container = st.container()

gold_tab, silver_tab, platinum_tab, palladium_tab = metals_container.tabs(
    [
        'Gold',
        'Silver',
        'Platinum',
        'Palladium'
    ]
)

with gold_tab:
    
    title_container = gold_tab.container()
    title_container.markdown("# Here is some info about gold")

    #вкладки с выбором периода
    day_tab, week_tab, month_tab, year_tab = gold_tab.tabs(
        [
        'Day',
        'Week',
        'Month',
        'Year'
        ]
    )

    #график и инфа в зависимости от выбранного периода
    with day_tab:
        show_info(day_tab)

    with week_tab:
        show_info(week_tab)

    with month_tab:
        show_info(month_tab)

    with year_tab:
        show_info(year_tab)
    
    #доп инфа
    additional_info_container = st.container(border=True)
    additional_info_text_container = additional_info_container.container(border=True)
    additional_info_text_container.write('blah blah blah blah')

with silver_tab:
    
    title_container = silver_tab.container()
    title_container.markdown("# Here is some info about silver")

    #вкладки с выбором периода
    day_tab, week_tab, month_tab, year_tab = silver_tab.tabs(
        [
        'Day',
        'Week',
        'Month',
        'Year'
        ]
    )

    #график и инфа в зависимости от выбранного периода
    with day_tab:
        show_info(day_tab)

    with week_tab:
        show_info(week_tab)

    with month_tab:
        show_info(month_tab)

    with year_tab:
        show_info(year_tab)

    #доп инфа
    additional_info_container = st.container(border=True)
    additional_info_text_container = additional_info_container.container(border=True)
    additional_info_text_container.write('blah blah blah blah')

with platinum_tab:
    
    title_container = platinum_tab.container()
    title_container.markdown("# Here is some info about platunim")

    #вкладки с выбором периода
    day_tab, week_tab, month_tab, year_tab = platinum_tab.tabs(
        [
        'Day',
        'Week',
        'Month',
        'Year'
        ]
    )

    #график и инфа в зависимости от выбранного периода
    with day_tab:
        show_info(day_tab)

    with week_tab:
        show_info(week_tab)

    with month_tab:
        show_info(month_tab)

    with year_tab:
        show_info(year_tab)

    #доп инфа
    additional_info_container = st.container(border=True)
    additional_info_text_container = additional_info_container.container(border=True)
    additional_info_text_container.write('blah blah blah blah')

with palladium_tab:
    
    title_container = palladium_tab.container()
    title_container.markdown("# Here is some info about palladium")

    #вкладки с выбором периода
    day_tab, week_tab, month_tab, year_tab = palladium_tab.tabs(
        [
        'Day',
        'Week',
        'Month',
        'Year'
        ]
    )

    #график и инфа в зависимости от выбранного периода
    with day_tab:
        show_info(day_tab)

    with week_tab:
        show_info(week_tab)

    with month_tab:
        show_info(month_tab)

    with year_tab:
        show_info(year_tab)

    #доп инфа
    additional_info_container = st.container(border=True)
    additional_info_text_container = additional_info_container.container(border=True)
    additional_info_text_container.write('blah blah blah blah')

#ссылки на покупку
st.markdown('# Buy metal in banks')
sber_link, tinkoff_link, vtb_link, rsh_link = st.columns(4)
sber_link.link_button('Sber', 'https://www.sberbank.com/ru/person/metall', use_container_width=True)
tinkoff_link.link_button('T-Bank', 'https://www.tbank.ru/invest/promo/gold/?ysclid=m9mn69syar893212723', use_container_width=True)
vtb_link.link_button('VTB', 'https://www.vtb.ru/personal/vklady-i-scheta/obezlichennyj-metallicheskij-schet/?ysclid=m9mn80q731797510826', use_container_width=True)
rsh_link.link_button('Rosselhoz', 'https://www.rshb.ru/natural/investments/metal-gold', use_container_width=True)
