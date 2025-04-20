import pandas as pd
import json
from tqdm import tqdm

def format_input_text(row, metal_name, date_format='%Y-%m-%d'):
    if isinstance(row.name, pd.Timestamp):
        date_str = row.name.strftime(date_format)
    elif 'timestamp' in row and pd.notna(row['timestamp']):
         if isinstance(row['timestamp'], pd.Timestamp):
             date_str = row['timestamp'].strftime(date_format)
         else:
            date_str = pd.to_datetime(row['timestamp']).strftime(date_format)

    ohlcv_text = (
        f"Open={row.get('open', 'N/A')}, "
        f"High={row.get('high', 'N/A')}, "
        f"Low={row.get('low', 'N/A')}, "
        f"Close={row.get('close', 'N/A')}, "
        f"Volume={row.get('volume', 'N/A')}"
    )

    news_text = row.get('headlines')
    news_text = ' '.join(str(news_text).split())

    inp_metal_name = metal_name[0].upper() + metal_name[1:]
    input_text = f"Дата: {date_str}. {inp_metal_name} OHLCV: {ohlcv_text}. Новости дня: {news_text}"
    return input_text


def convert_csv_to_jsonl(csv_filepath, jsonl_filepath, metal_name, label_column='label'):
    df = pd.read_csv(csv_filepath, index_col='timestamp', parse_dates=True, sep=';')
    print(f"Загружен CSV файл: {csv_filepath}. Строк: {len(df)}")

    original_count = len(df)
    df.dropna(subset=[label_column], inplace=True)
    filtered_count = len(df)
    print(f"Удалено {original_count - filtered_count} строк без метки в колонке '{label_column}'. Осталось строк: {filtered_count}")

    if filtered_count == 0:
        print("Нет данных для конвертации после фильтрации.")
        return

    with open(jsonl_filepath, 'w', encoding='utf-8') as outfile:
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Конвертация в JSONL"):
            input_text = format_input_text(row, metal_name)

            output_label = row[label_column]

            record = {
                "input_text": input_text,
                "output_label": output_label
            }

            json_string = json.dumps(record, ensure_ascii=False)

            outfile.write(json_string + '\n')

    print(f"Конвертация завершена. Результат сохранен в: {jsonl_filepath}")


if __name__ == "__main__":
    metal_name = 'palladium'

    convert_csv_to_jsonl(f'../datasets/labeled_dataset/labeled_{metal_name}_data.csv', f'json_data_{metal_name}', metal_name = metal_name)