import os
import gc
import sys

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from evaluate import load


def cleanup() -> None:
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()


def get_n_layers(model_name: str) -> int:
    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, 'num_hidden_layers'):
        return config.num_hidden_layers
    elif hasattr(config, 'n_layers'):
        return config.n_layers


def truncate_text(text: str, tokenizer: AutoTokenizer, max_length: int = 512) -> (str, bool):
    tokens = tokenizer(text, max_length=max_length, truncation=True, return_overflowing_tokens=True)
    truncated_text = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
    truncated = len(tokens['input_ids'])!=1
    return truncated_text, truncated


def compute_bertscore(df: pd.DataFrame, cur_path: str, lang_type: str) -> None:
    if lang_type == 'multilingual':
        model_names = ['bert-base-multilingual-cased',
                       'xlm-mlm-100-1280', 
                       'distilbert-base-multilingual-cased', 
                       'xlm-roberta-base', 
                       'xlm-roberta-large', 
                       'facebook/mbart-large-cc25', 
                       'facebook/mbart-large-50', 
                       'facebook/mbart-large-50-many-to-many-mmt', 
                       'google/mt5-small', 
                       'google/mt5-base', 
                       'google/mt5-large', 
                       'google/mt5-xl', 
                       'google/byt5-small', 
                       'google/byt5-base', 
                       'google/byt5-large', 
                       'microsoft/mdeberta-v3-base']
    elif lang_type == 'ru':
        model_names = ['DeepPavlov/rubert-base-cased',
                       'ai-forever/ruBert-base',
                       'ai-forever/ruBert-large',
                       'ai-forever/ruRoberta-large',
                       'ai-forever/ru-en-RoSBERTa']
        
    for model in model_names:
        print(f'MODEL: {model}')
        num_layers = get_n_layers(model)
        tokenizer = AutoTokenizer.from_pretrained(model)
        
        pred_truncated_flags = []
        ref_truncated_flags = []
        truncated_pred_texts = []
        truncated_ref_texts = []

        for index, row in df.iterrows():
            pred_text = row['pred']
            ref_text = row['output']
            
            if pd.isnull(ref_text) or pd.isnull(pred_text):
                pred_truncated_flags.append(False)
                ref_truncated_flags.append(False)
                truncated_pred_texts.append(pred_text)
                truncated_ref_texts.append(ref_text)
            else:
                truncated_pred_text, pred_truncated = truncate_text(pred_text, tokenizer)
                truncated_ref_text, ref_truncated = truncate_text(ref_text, tokenizer)
                
                pred_truncated_flags.append(pred_truncated)
                ref_truncated_flags.append(ref_truncated)
                truncated_pred_texts.append(truncated_pred_text)
                truncated_ref_texts.append(truncated_ref_text)
        
        df[f'pred_truncated_{model}'] = pred_truncated_flags
        df[f'ref_truncated_{model}'] = ref_truncated_flags
        
        for layer in tqdm(range(num_layers)):
            bertscores = []
            for index, row in df.iterrows():
                pred_text = truncated_pred_texts[index]
                ref_text = truncated_ref_texts[index]
                if pd.isnull(ref_text) or pd.isnull(pred_text):
                    bertscores.append(0) 
                else:
                    try:
                        bert_score = bertscore.compute(predictions=[pred_text], 
                                                       references=[ref_text], 
                                                       lang='ru', 
                                                       model_type=model, 
                                                       num_layers=layer)
                    except RuntimeError:  # usually, it is out-of-memory
                        cleanup()
                        bert_score = bertscore.compute(predictions=[pred_text], 
                                                       references=[ref_text], 
                                                       lang='ru', 
                                                       model_type=model, 
                                                       num_layers=layer)
                    bertscores.append(bert_score['f1'][0])
            df[f"{model}_layer_{layer}"] = bertscores
            df.to_csv(cur_path, index=False) # save dataframe after each layer to avoid data loss

            
if len(sys.argv) == 3:
    data_folder = sys.argv[1] #'yandexgpt' or 'gigachat'
    model_lang = sys.argv[2] #'multilingual' or 'ru'
else:
    if len(sys.argv) != 3:
        print("Ошибка. Вы должны ввести название папки с данными ('yandexgpt' или 'gigachat') и языковой тип моделей ('multilingual' или 'ru')")
        sys.exit(1)

file_path = os.path.abspath(__file__)

PROJECT_PATH = os.path.dirname(os.path.dirname(file_path))
DATA_PATH = os.path.join(PROJECT_PATH, data_folder)
if model_lang == 'multilingual':
    BERTSCORE_PATH = os.path.join(PROJECT_PATH, 'computed_bertscore')
elif model_lang == 'ru':
    BERTSCORE_PATH = os.path.join(PROJECT_PATH, 'computed_ru_bertscore')
bertscore = load("bertscore")

for folder in os.listdir(DATA_PATH):
    dataset_path = DATA_PATH+'/'+folder
    if os.path.isdir(dataset_path):
        for csv_file in os.listdir(dataset_path):
            if csv_file.endswith(".csv"):
                print(f'------------------FILE: {csv_file}------------------')
                current_path = BERTSCORE_PATH+'/'+data_folder+'/'+folder
                os.makedirs(current_path, exist_ok=True)
                data_df = pd.read_csv(dataset_path+'/'+csv_file)
                compute_bertscore(data_df, current_path+'/'+csv_file, model_lang)
