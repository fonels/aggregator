from tqdm import tqdm

def split_jsonl(input_filepath, train_filepath, validation_filepath, split_ratio=0.85):

    print(f"Подсчет строк в файле: {input_filepath}...")
    with open(input_filepath, 'r', encoding='utf-8') as infile:
        total_lines = sum(1 for _ in infile)

    print(f"Всего строк найдено: {total_lines}")

    split_line_num = int(total_lines * split_ratio)

    train_lines_count = split_line_num
    validation_lines_count = total_lines - split_line_num

    print(f"Запись обучающей выборки в {train_filepath}...")
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(train_filepath, 'w', encoding='utf-8') as outfile_train:

        for i, line in enumerate(tqdm(infile, total=train_lines_count, desc="Запись train")):
            if i < train_lines_count:
                outfile_train.write(line)
            else:
                break

    print(f"Запись валидационной выборки в {validation_filepath}...")
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(validation_filepath, 'w', encoding='utf-8') as outfile_val:

         line_iterator = enumerate(infile)
         for _ in tqdm(range(train_lines_count), total=train_lines_count, desc="Пропуск train строк", leave=False):
             try:
                 next(line_iterator)
             except StopIteration:
                 break

         for i, line in tqdm(line_iterator, total=validation_lines_count, initial=train_lines_count, desc="Запись validation"):
             outfile_val.write(line)

    print("Разделение файлов по количеству строк завершено успешно.")


if __name__ == "__main__":
    metal_name = 'palladium'
    split_jsonl(input_filepath = f'../datasets/labeled_dataset/json_data_{metal_name}.jsonl', train_filepath = f'../datasets/labeled_dataset/train/json_train_{metal_name}.jsonl', \
                validation_filepath = f'../datasets/labeled_dataset/valid/json_valid_{metal_name}.jsonl')