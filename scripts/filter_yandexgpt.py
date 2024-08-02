import os

import pandas as pd


file_path = os.path.abspath(__file__)

PROJECT_PATH = os.path.dirname(os.path.dirname(file_path))
YANDEXGPT_PATH = os.path.join(PROJECT_PATH, 'yandexgpt')

for folder in os.listdir(YANDEXGPT_PATH):
    data_path = YANDEXGPT_PATH+'/'+folder
    if os.path.isdir(data_path):
        for csv_file in os.listdir(data_path):
            if csv_file.endswith(".csv"):
                yandexgpt_df = pd.read_csv(data_path+'/'+csv_file)
                if not yandexgpt_df.empty:
                    yandexgpt_df = yandexgpt_df.dropna(subset=['pred'])
                    yandexgpt_df = yandexgpt_df.loc[(yandexgpt_df['pred'] != '') & (yandexgpt_df['pred'] != 'FAILED') & (yandexgpt_df['output'] != '.')]
                    yandexgpt_df.to_csv(data_path+'/'+csv_file, index=False)
