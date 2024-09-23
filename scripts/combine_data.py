import os 
import sys

import pandas as pd


file_path = os.path.abspath(__file__)

if len(sys.argv) == 2:
    data_folder = sys.argv[1] #'yandexgpt' or 'gigachat'
else:
    if len(sys.argv) != 3:
        print("Ошибка. Вы должны ввести название папки с данными ('yandexgpt' или 'gigachat')")
        sys.exit(1)

PROJECT_PATH = os.path.dirname(os.path.dirname(file_path))
DATA1_PATH = os.path.join(PROJECT_PATH, f'ru-bert-score/computed_ru_bertscore/{data_folder}') #путь до результатов bert, roberta
DATA2_PATH = os.path.join(PROJECT_PATH, f'bertscore/computed_ru_bertscore/{data_folder}') #путь до результатов t5, longformer

COMBINED_PATH = os.path.join(PROJECT_PATH, f'ru-bert-score/computed_ru_bertscore/{data_folder}')

for folder in os.listdir(DATA1_PATH):
    current_path = DATA1_PATH+'/'+folder
    if os.path.isdir(current_path):
        for csv_file in os.listdir(current_path):
            if csv_file.endswith(".csv"):
                print(f'FILE: {csv_file}')
                
                bertscore1_df = pd.read_csv(current_path+'/'+csv_file)
                bertscore2_df = pd.read_csv(DATA2_PATH+'/'+folder+'/'+csv_file)

                if csv_file == 'science_dataset.csv' and data_folder == 'gigachat':
                    input_column = 'filename'
                else:
                    input_column = 'input'
                
                if csv_file == 'preds.csv' and data_folder == 'yandexgpt':
                    del_column_list = [input_column, 'output', 'pred', 'dir_name']
                else:
                    del_column_list = [input_column, 'output', 'pred']
                for del_column in del_column_list:
                    del bertscore2_df[del_column]

                df_combined = bertscore1_df.join(
                    bertscore2_df
                )
                os.makedirs(COMBINED_PATH+'/'+folder, exist_ok=True)
                df_combined.to_csv(COMBINED_PATH+'/'+folder+'/'+csv_file, index=False)