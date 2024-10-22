import os
import gc
import sys

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
from evaluate import load

try:
    from bertscore import bert_score
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from bertscore import bert_score
    

def cleanup() -> None:
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()


def get_n_layers_and_architectures(model_name: str) -> int:
    config = AutoConfig.from_pretrained(model_name)
    
    if hasattr(config, 'num_hidden_layers') and hasattr(config, 'architectures'):
        return config.num_hidden_layers, config.architectures[0]
    elif hasattr(config, 'n_layers') and hasattr(config, 'architectures'):
        return config.n_layers, config.architectures[0]


def truncate_text(text: str, tokenizer: AutoTokenizer, model_name: str, max_length: int = 512) -> (str, bool):
    tokens = tokenizer(text, max_length=max_length, truncation=True, return_overflowing_tokens=True, add_special_tokens=False)
    if model_name == 'xlm-mlm-100-1280':
        truncated_text = tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)
    else:
        truncated_text = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
    truncated = len(tokens['input_ids']) != 1
    return truncated_text, truncated


def call_bertscore(architectures, references, predictions, tokenizer, model, layer):
    if architectures == 'LongformerForMaskedLM':
        bertscores = bert_score(references, predictions, (tokenizer, model), 8, layer, True) # use_global_attention = True if model - longformer
    elif architectures == 'T5ForConditionalGeneration':
        bertscores = bert_score(references, predictions, (tokenizer, model.encoder), 8, layer)
    else:
        bertscores = bert_score(references, predictions, (tokenizer, model), 8, layer)
    return bertscores


def manual_bertscore(df: pd.DataFrame, model_name: str, cur_path: str) -> None:
    if 'ruT5' in model_name:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name) 
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name) 

    num_layers, architectures = get_n_layers_and_architectures(model_name)

    for layer in tqdm(range(num_layers)):
        references = list(df['output'])
        predictions = list(df['pred'])
        try:
            bertscores = call_bertscore(architectures, references, predictions, tokenizer, model, layer)
        except RuntimeError:  # usually, it is out-of-memory
            cleanup()
            bertscores = call_bertscore(architectures, references, predictions, tokenizer, model, layer)
        df[f"{model_name}_layer_{layer}"] = bertscores
        df.to_csv(cur_path, index=False) # save dataframe after each layer to avoid data loss


def auto_bertscore(df: pd.DataFrame, model_name: str, cur_path: str) -> None:   
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_layers, _ = get_n_layers_and_architectures(model_name)

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
            truncated_pred_text, pred_truncated = truncate_text(pred_text, tokenizer, model_name)
            truncated_ref_text, ref_truncated = truncate_text(ref_text, tokenizer, model_name)
            
            pred_truncated_flags.append(pred_truncated)
            ref_truncated_flags.append(ref_truncated)
            truncated_pred_texts.append(truncated_pred_text)
            truncated_ref_texts.append(truncated_ref_text)
    
    df[f'pred_truncated_{model_name}'] = pred_truncated_flags
    df[f'ref_truncated_{model_name}'] = ref_truncated_flags
    
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
                                                    model_type=model_name, 
                                                    num_layers=layer)
                except RuntimeError:  # usually, it is out-of-memory
                    cleanup()
                    bert_score = bertscore.compute(predictions=[pred_text], 
                                                    references=[ref_text], 
                                                    lang='ru', 
                                                    model_type=model_name, 
                                                    num_layers=layer)
                bertscores.append(bert_score['f1'][0])
        df[f"{model_name}_layer_{layer}"] = bertscores
        df.to_csv(cur_path, index=False) # save dataframe after each layer to avoid data loss


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
                       'microsoft/mdeberta-v3-base'
                       ]
    elif lang_type == 'ru':
        model_names = ['DeepPavlov/rubert-base-cased',
                       'ai-forever/ruBert-base',
                       'ai-forever/ruBert-large',
                       'ai-forever/ruRoberta-large',
                       'ai-forever/ru-en-RoSBERTa',
                       'bond005/rubert-entity-embedder',
                       'ai-forever/ruSciBERT',
                       'cointegrated/rubert-tiny2',
                       'kazzand/ru-longformer-tiny-16384',
                       'kazzand/ru-longformer-base-4096',
                       'kazzand/ru-longformer-large-4096',
                       'ai-forever/ruT5-base',
                       'ai-forever/ruT5-large',
                       'ai-forever/FRED-T5-large'
                       ]

    for model_name in model_names:
        print(f'MODEL: {model_name}')
        if lang_type == 'ru' and ('T5' in model_name or 'longformer' in model_name):
            df['output'].fillna('', inplace=True)
            df['pred'].fillna('', inplace=True)
            manual_bertscore(df, model_name, cur_path)     
        else:
            auto_bertscore(df, model_name, cur_path)


if __name__ == '__main__':            
    if len(sys.argv) == 3:
        data_folder = sys.argv[1] #'yandexgpt' or 'gigachat'
        model_lang = sys.argv[2] #'multilingual' or 'ru'
    else:
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
