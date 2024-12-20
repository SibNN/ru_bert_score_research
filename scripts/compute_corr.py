import os 
import sys

import pandas as pd
from scipy import stats


def clean_text(text):
    text = text.strip()
    text = text.replace('\r', '')
    return text


def compute_correlation(current_path, csv_file, data_folder, folder, model_score_folder, corr_current_path):
    bertscore_df = pd.read_csv(current_path+'/'+csv_file)
    bertscore_df = bertscore_df.dropna()

    if csv_file == 'science_dataset.csv' and data_folder == 'gigachat':
        input_column = 'filename'
    else:
        input_column = 'input'

    bertscore_columns = [column for column in list(bertscore_df.keys()) if 'layer' in column]

    bertscore_df['average_score'] = bertscore_df[bertscore_columns].mean(axis=1)

    best_outputs = bertscore_df.loc[bertscore_df.groupby(input_column)['average_score'].idxmax()] #save only those ground truth outputs that have the highest average score according to bertscore
    best_outputs = best_outputs.drop(columns=['average_score'])

    best_outputs[input_column] = best_outputs[input_column].apply(clean_text)
    best_outputs['output'] = best_outputs['output'].apply(clean_text)

    model_df = pd.read_csv(SCORE_PATH+'/'+folder+'/'+csv_file)
    model_df = model_df.drop_duplicates()
    model_df = model_df.dropna()

    model_df[input_column] = model_df[input_column].apply(clean_text)
    model_df['output'] = model_df['output'].apply(clean_text)

    if csv_file == 'vacancies.csv' and data_folder == 'gigachat':
        model_score_columns = ['scores_1','score_2']
    elif data_folder == 'gigachat':
        model_score_columns = ['scores_1','score_2','score_3']
    elif data_folder == 'yandexgpt':
        model_score_columns = ['gigachat_scores_1','gigachat_scores_2','gigachat_scores_3']
    
    for cur_column in model_score_columns:
        model_df = model_df[model_df[cur_column] != 'FAILED']

    model_df[f'avg_{model_score_folder}'] = model_df[model_score_columns].astype(float).mean(axis=1)
    model_df = model_df.drop(columns=model_score_columns)

    merged_df = pd.merge(best_outputs, model_df, on=[input_column,'output'], how='inner')

    pearson_results = []
    kendall_results = []
    for bertscore_column in bertscore_columns:
        try:
            bertscore_list = merged_df[bertscore_column].values
            model_score_list = merged_df[f'avg_{model_score_folder}'].values
            pearson_results.append(stats.pearsonr(bertscore_list, model_score_list).statistic)
            kendall_results.append(stats.kendalltau(bertscore_list, model_score_list).statistic)
        except ValueError as e:
            print(e)

    res_df = pd.DataFrame({'model_layer': bertscore_columns,
                        'pearson': pearson_results,
                        'kendall': kendall_results})

    os.makedirs(corr_current_path, exist_ok=True)
    res_df.to_csv(corr_current_path+'/'+csv_file, index=False)


if __name__ == '__main__': 
    if len(sys.argv) == 3:
        data_folder = sys.argv[1] #'yandexgpt' or 'gigachat'
        model_lang = sys.argv[2] #'multilingual' or 'ru'
    else:
        raise ValueError("Неправильное количество аргументов. Ожидалось 2 аргумента: название папки с данными ('yandexgpt' или 'gigachat') и тип модели ('multilingual' или 'ru').")
        
    DATA_FOLDER2MODEL_SCORE_FOLDER = {
        'yandexgpt': 'gigachat_score',
        'gigachat': 'yandex_scores'
    }
    
    MODEL_LANG2FOLDER_NAME = {
        'multilingual': ['computed_bertscore', 'correlations'],
        'ru': ['computed_ru_bert_bertscore', 'ru_correlations']
    }
    
    assert data_folder in DATA_FOLDER2MODEL_SCORE_FOLDER, \
        f'Only {DATA_FOLDER2MODEL_SCORE_FOLDER.keys()} are accepted, but {data_folder} was passed'
    
    assert model_lang in MODEL_LANG2FOLDER_NAME, \
        f'Only {MODEL_LANG2FOLDER_NAME.keys()} are accepted, but {model_lang} was passed'
    
    model_score_folder = DATA_FOLDER2MODEL_SCORE_FOLDER[data_folder]

    file_path = os.path.abspath(__file__)

    PROJECT_PATH = os.path.dirname(os.path.dirname(file_path))
    folder_names = MODEL_LANG2FOLDER_NAME[model_lang]

    BERTSCORE_PATH = os.path.join(PROJECT_PATH, folder_names[0])
    CORR_PATH = os.path.join(PROJECT_PATH, folder_names[1])

    SCORE_PATH = os.path.join(PROJECT_PATH, model_score_folder)

    for folder in os.listdir(BERTSCORE_PATH+'/'+data_folder):
        current_path = BERTSCORE_PATH+'/'+data_folder+'/'+folder
        corr_current_path = CORR_PATH+'/'+data_folder+'/'+folder
        if os.path.isdir(current_path):
            for csv_file in os.listdir(current_path):
                if csv_file.endswith(".csv"):
                    if csv_file!='test.csv':
                        print(f'FILE: {csv_file}')
                        compute_correlation(current_path, csv_file, data_folder, folder, model_score_folder, corr_current_path)
