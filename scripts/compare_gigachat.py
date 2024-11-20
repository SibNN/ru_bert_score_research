import os

import pandas as pd
from tqdm import tqdm
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models import GigaChat
import yaml

from scripts.paths import CONFIG_PATH


config = yaml.safe_load(open(CONFIG_PATH))


def get_score(prompt: str) -> int:
    messages = [
        SystemMessage(
            content="Ты система, которая оценивает качество сгенерированного текста."
            ),
            HumanMessage(content=prompt)
    ]

    res = chat(messages)
    if res.response_metadata['finish_reason'] == 'blacklist':
        score = -1
    else:
        score = res.content
    return int(score)
    
    
def gigachat_scorer(df: pd.DataFrame, cur_path: str) -> None:
    new_columns = ["gigachat_scores_1", 
                   "gigachat_scores_2", 
                   "gigachat_scores_3"]

    dataset2task_desc = {
            'dialogsum_ru': 'Генерация краткого пересказа диалога',
            'reviews_russian': 'Генерация краткого пересказа отзыва об отеле',
            'ru_simple_sent_eval': 'Генерация текста, более простого для понимания',
            'science_summarization_dataset': 'Генерация краткого пересказа научной статьи',
            'telegram-financial-sentiment-summarization': 'Генерация краткого пересказа новостного текста',
            'yandex_jobs': 'Генерация названия должности по описанию вакансии'
    }
    task = dataset2task_desc[os.path.basename(os.path.dirname(cur_path))]
    
    df = df.reindex(columns = df.columns.tolist() + new_columns)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        pred_text = row['pred']
        ref_text = row['output']
        
        prompts = [
            f'Оцени качество сгенерированного текста по 5-балльной шкале, насколько он соответствует эталонному тексту. \
               Выдай только одно число от 1 до 5, где 1 - сгенерированный текст не соответствует эталонному, 5 - сгенерированный текст полностью соответствует эталонному.\n \
               Эталонный текст: {ref_text}\n \
               Сгенерированный текст: {pred_text}',
            f'Оцени, насколько сгенерированный текст совпадает с эталонным, по шкале от 1 до 5. \
               Укажите одно число, где 1 - это полное несоответствие, а 5 - полное совпадение.\n \
               Задача генерации: {task}\n "\
               Эталонный текст: {ref_text}\n \
               Сгенерированный текст: {pred_text}',
            f'Сравни два предложенных сгенерированных текста и поставь оценку схожести по смыслу по 5-балльной шкале, где 1 - абсолютно не похожи, 5 - абсолютно похожи.\n \
               Эталонный текст: {ref_text}\n \
               Сгенерированный текст: {pred_text}'
        ]
            
        for i, prompt in enumerate(prompts):
            if pd.isnull(ref_text) or pd.isnull(pred_text):
                df.at[index, f"gigachat_scores_{i+1}"] = 0
            else:
                df.at[index, f"gigachat_scores_{i+1}"] = int(get_score(prompt))

        df.to_csv(cur_path, index=False)
     
        
file_path = os.path.abspath(__file__)

data_folder = 'yandexgpt'

PROJECT_PATH = os.path.dirname(os.path.dirname(file_path))
DATA_PATH = os.path.join(PROJECT_PATH, data_folder)
SCORE_PATH = os.path.join(PROJECT_PATH, 'gigachat_score')

chat = GigaChat(credentials=config['credentials'], scope=config['scope'], verify_ssl_certs=False)

for folder in os.listdir(DATA_PATH):
    data_path = DATA_PATH+'/'+folder
    if os.path.isdir(data_path):
        for csv_file in os.listdir(data_path):
            if csv_file.endswith(".csv"):
                print(f'------------------FILE: {csv_file}------------------')
                current_path = SCORE_PATH+'/'+folder
                os.makedirs(current_path, exist_ok=True)
                data_df = pd.read_csv(data_path+'/'+csv_file)
                gigachat_scorer(data_df, current_path+'/'+csv_file)        
