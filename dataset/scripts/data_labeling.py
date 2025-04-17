import pandas as pd
import numpy as np

def load_data(filepath):
    df = pd.read_csv(filepath, sep = ';', parse_dates=['timestamp'])

    if 'timestamp' not in df.columns or 'close' not in df.columns:
        print(f"Ошибка: Файл {filepath} должен содержать колонки 'timestamp' и 'close'.")
        return None

    df.set_index('timestamp', inplace=True)

    df.sort_index(inplace=True)

    print(f"Данные успешно загружены. Период: {df.index.min()} - {df.index.max()}. Записей: {len(df)}")
    return df

def add_fpm_labels(df, n_days = 5, threshold_buy = 1.0, threshold_sell = 1.0):

    future_close = df['close'].shift(-n_days)

    df['future_percent_change'] = (future_close - df['close']) / df['close'] * 100

    cond_buy = df['future_percent_change'] > threshold_buy
    cond_sell = df['future_percent_change'] < -threshold_sell

    conditions = [cond_buy, cond_sell]
    choices = ['Buy', 'Sell']
    df['label'] = np.select(conditions, choices, default='Hold')

    df.loc[df['future_percent_change'].isna(), 'label'] = np.nan

    print("Разметка FPM завершена.")
    print("Распределение меток:")
    print(df['label'].value_counts(dropna=False))

    return df

if __name__ == "__main__":
    metal_name = 'palladium'
    start_dataset = load_data(f'../datasets/final_{metal_name}_data.csv')
    labeled_dataset = add_fpm_labels(start_dataset)
    labeled_dataset.reset_index().to_csv(f'../datasets/labeled_{metal_name}_data.csv', index=False, sep=';')