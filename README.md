# Aggregator: Аналитический дашборд для рынка драгоценных металлов

**Aggregator** — это комплексное приложение для сбора, обработки и анализа финансовых новостей и данных, связанных с рынком драгоценных металлов (золото, серебро, платина, палладий). Проект использует дообученные (fine-tuned) языковые модели для генерации инвестиционных рекомендаций (Buy/Sell/Hold) и представляет результаты в интерактивном веб-интерфейсе, созданном с помощью Streamlit.

## 🚀 Основные возможности

* **Автоматический сбор данных**: Ежедневный запуск конвейера, который собирает исторические котировки (OHLCV) с помощью `yfinance` и парсит новостные заголовки с архива The Wall Street Journal.
* **Анализ с помощью ИИ**: Для каждого металла используется собственная версия дообученной модели `zephyr-7b-beta`, которая анализирует дневные котировки и новостной фон для генерации прогноза на следующую торговую неделю.
* **Централизованное хранение**: Все собранные данные сохраняются в базу данных PostgreSQL для быстрого доступа и анализа.
* **Интерактивный дашборд**: Веб-приложение на Streamlit отображает исторические графики цен, актуальные новости и позволяет запрашивать прогнозы у моделей в реальном времени.

## 🛠️ Стек технологий

* **Backend**: FastAPI, Uvicorn 
* **Frontend (дашборд)**: Streamlit 
* **Машинное обучение**:
    * PyTorch 
    * Hugging Face Transformers (для работы с моделью `zephyr-7b-beta`) 
    * PEFT (Parameter-Efficient Fine-Tuning) для дообучения с помощью LoRA 
    * TRL (Transformer Reinforcement Learning) для Supervised Fine-tuning 
* **Сбор данных**:
    * yfinance (финансовые данные) 
    * Beautiful Soup & Requests (парсер новостей) 
* **База данных**: PostgreSQL (с драйвером `psycopg2`) 
* **Трекинг экспериментов**: Weights & Biases (`wandb`) 
* **Обработка данных**: Pandas, NumPy 

## 📂 Структура проекта

```
aggregator/
├── .streamlit/
│   ├── config.toml         # Конфигурация Streamlit
│   └── ochita.py           # Главный скрипт Streamlit-приложения (frontend)
├── data_pipeline/
│   ├── financial_data_fetcher.py # Скрипт для сбора финансовых данных через yfinance
│   └── news_fetcher.py     # Скрипт для парсинга новостей с wsj
├── dataset/
│   ├── datasets/           # Размеченные данные для обучения (train/valid)
│   └── scripts/            # Скрипты для обработки данных
├── model/
│   ├── model_train.py      # Скрипт для дообучения моделей на каждый металл
│   └── gold/silver/...     # Директории с весами обученных LoRA-адаптеров
├── .gitignore
├── LICENSE
[cite_start]├── requirements.txt        # Зависимости проекта 
├── run_daily_update.py     # Главный скрипт для запуска ежедневного обновления данных в БД
└── server.py               # FastAPI-сервер (backend), предоставляющий API для моделей и данных
```

## ⚙️ Установка и запуск

### 1. Предварительные требования
* Установленный Python 3.8+
* Доступ к работающему серверу PostgreSQL.

### 2. Установка

```bash
# Клонируйте репозиторий
git clone https://github.com/fonels/aggregator.git
cd aggregator

# Создайте и активируйте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Для Windows: venv\Scripts\activate

# Установите зависимости
pip install -r requirements.txt
```

### 3. Настройка
Перед запуском убедитесь, что в файле `data_pipeline/config.py` указаны корректные данные для подключения к вашей базе данных PostgreSQL (`DB_CONFIG`).

### 4. Запуск приложения

Приложение состоит из двух основных частей: backend-сервера FastAPI и frontend-приложения Streamlit. Их нужно запускать в отдельных терминалах.

**А. Запустите Backend-сервер:**
```bash
uvicorn server:app --reload
```
Сервер будет доступен по адресу `http://localhost:8000`.

**Б. Запустите Frontend-приложение:**
```bash
streamlit run .streamlit/ochita.py
```
Веб-интерфейс откроется в вашем браузере по адресу, указанному в терминале.

### 5. Запуск конвейера данных

Для первоначального наполнения базы данных или для ежедневного обновления запустите:
```bash
python run_daily_update.py
```
Этот скрипт соберёт данные за последние сутки и загрузит их в PostgreSQL.

## 📈 Использование

1.  Откройте приложение Streamlit в браузере.
2.  На главной странице вы увидите вкладки для каждого металла: **Gold, Silver, Platinum, Palladium**.
3.  На каждой вкладке можно переключаться между периодами (**Week, Month, Year**) для просмотра графика цен.
4.  Ниже графиков расположен блок с последними финансовыми новостями.
5.  Для получения прогноза на следующую неделю, нажмите кнопку **"Make prediction"**. Приложение отправит запрос на backend, где модель сгенерирует рекомендацию (Buy/Sell/Hold) и её обоснование, которые затем отобразятся на странице.

## 📜 Лицензия

Проект распространяется под лицензией MIT. См. файл `LICENSE` для подробностей.

---
*Отказ от ответственности: Информация, представленная на этом сайте, предназначена исключительно для информационных целей и не является финансовой, инвестиционной или иной профессиональной консультацией.*

-----

# Aggregator: Analytical Dashboard for the Precious Metals Market

**Aggregator** is a comprehensive application for collecting, processing, and analyzing financial news and data related to the precious metals market (gold, silver, platinum, palladium). The project utilizes fine-tuned language models to generate investment recommendations (Buy/Sell/Hold) and presents the results in an interactive web interface built with Streamlit.

## 🚀 Key Features

  * **Automated Data Collection**: A daily pipeline gathers historical quotes (OHLCV) using `yfinance` and parses news headlines from The Wall Street Journal's archives.
  * **AI-Powered Analysis**: Each metal has its own fine-tuned version of the `zephyr-7b-beta` model, which analyzes daily quotes and news to generate a forecast for the upcoming trading week.
  * **Centralized Storage**: All collected data is saved to a PostgreSQL database for quick access and analysis.
  * **Interactive Dashboard**: A Streamlit web application displays historical price charts, current news, and allows users to request real-time forecasts from the models.

## 🛠️ Technology Stack

  * **Backend**: FastAPI, Uvicorn
  * **Frontend (Dashboard)**: Streamlit
  * **Machine Learning**:
      * PyTorch
      * Hugging Face Transformers (for the `zephyr-7b-beta` model)
      * PEFT (Parameter-Efficient Fine-Tuning) for LoRA fine-tuning
      * TRL (Transformer Reinforcement Learning) for Supervised Fine-tuning
  * **Data Collection**:
      * yfinance (financial data)
      * Beautiful Soup & Requests (news parser)
  * **Database**: PostgreSQL (with the `psycopg2` driver)
  * **Experiment Tracking**: Weights & Biases (`wandb`)
  * **Data Processing**: Pandas, NumPy

## 📂 Project Structure

```
aggregator/
├── .streamlit/
│   ├── config.toml         # Streamlit configuration
│   └── ochita.py           # Main script for the Streamlit frontend
├── data_pipeline/
│   ├── financial_data_fetcher.py # Script for fetching financial data via yfinance
│   └── news_fetcher.py     # Script for parsing news from wsj
├── dataset/
│   ├── datasets/           # Labeled data for training (train/valid)
│   └── scripts/            # Scripts for data processing
├── model/
│   ├── model_train.py      # Script for fine-tuning models for each metal
│   └── gold/silver/...     # Directories with trained LoRA adapter weights
├── .gitignore
├── LICENSE
├── requirements.txt        # Project dependencies
├── run_daily_update.py     # Main script to run daily data updates to the DB
└── server.py               # FastAPI backend server providing an API for models and data
```

## ⚙️ Installation and Setup

### 1\. Prerequisites

  * Python 3.8+ installed
  * Access to a running PostgreSQL server.

### 2\. Installation

```bash
# Clone the repository
git clone https://github.com/fonels/aggregator.git
cd aggregator

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3\. Configuration

Before running, ensure that the `data_pipeline/config.py` file contains the correct connection details for your PostgreSQL database (`DB_CONFIG`).

### 4\. Running the Application

The application consists of two main parts: a FastAPI backend server and a Streamlit frontend application. They need to be run in separate terminals.

**A. Start the Backend Server:**

```bash
uvicorn server:app --reload
```

The server will be available at `http://localhost:8000`.

**B. Start the Frontend Application:**

```bash
streamlit run .streamlit/ochita.py
```

The web interface will open in your browser at the address shown in the terminal.

### 5\. Running the Data Pipeline

To perform an initial population of the database or to run daily updates, execute:

```bash
python run_daily_update.py
```

This script will collect data for the last 24 hours and upload it to PostgreSQL.

## 📈 Usage

1.  Open the Streamlit application in your browser.
2.  On the main page, you will see tabs for each metal: **Gold, Silver, Platinum, and Palladium**.
3.  In each tab, you can switch between time periods (**Week, Month, Year**) to view the price chart.
4.  Below the charts, there is a block with the latest financial news.
5.  To get a forecast for the next week, click the **"Make prediction"** button. The application will send a request to the backend, where the model will generate a recommendation (Buy/Sell/Hold) and its justification, which will then be displayed on the page.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

-----

*Disclaimer: The information presented on this site is intended for informational purposes only and does not constitute financial, investment, or other professional advice.*