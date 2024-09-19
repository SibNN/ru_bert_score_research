import os
import sys
import gc

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig
from transformers import LongformerForMaskedLM, LongformerTokenizerFast, T5Tokenizer, T5ForConditionalGeneration

try:
    from bertscore import calculate_token_embeddings, bert_score
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from bertscore import calculate_token_embeddings, bert_score
    

def cleanup() -> None:
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()


def get_n_layers_and_architectures(model_name: str) -> int:
    config = AutoConfig.from_pretrained(model_name)
    
    if hasattr(config, 'num_hidden_layers') and (config, 'architectures'):
        return config.num_hidden_layers, config.architectures[0]
    elif hasattr(config, 'n_layers') and (config, 'architectures'):
        return config.n_layers, config.architectures[0]


def compute_bertscore(df: pd.DataFrame, cur_path: str) -> None:
    model_names = ['kazzand/ru-longformer-tiny-16384',
                    'kazzand/ru-longformer-base-4096',
                    'kazzand/ru-longformer-large-4096',
                    'ai-forever/ruT5-base',
                    'ai-forever/ruT5-large',
                    'ai-forever/FRED-T5-large'
                  ]
    df['output'].fillna('', inplace=True)
    df['pred'].fillna('', inplace=True)
    for model_name in model_names:
        if 'longformer' in model_name:
            tokenizer = LongformerTokenizerFast.from_pretrained(model_name)
            model = LongformerForMaskedLM.from_pretrained(model_name) 
        elif 'T5' in model_name:
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            
        print(f'MODEL: {model_name}')
        num_layers, architectures = get_n_layers_and_architectures(model_name)

        for layer in tqdm(range(num_layers)):
            references = list(df['output'])
            predictions = list(df['pred'])
            try:
                if architectures == 'LongformerForMaskedLM':
                    bertscores = bert_score(references, predictions, (tokenizer, model), 8, True) # use_global_attention = True if model - longformer
                elif architectures == 'T5ForConditionalGeneration':
                    bertscores = bert_score(references, predictions, (tokenizer, model.encoder), 8, layer)
                else:
                    bertscores = bert_score(references, predictions, (tokenizer, model), 8, layer)
            except RuntimeError:  # usually, it is out-of-memory
                cleanup()
                if architectures == 'LongformerForMaskedLM':
                    bertscores = bert_score(references, predictions, (tokenizer, model), 8, True) # use_global_attention = True if model - longformer
                elif architectures == 'T5ForConditionalGeneration':
                    bertscores = bert_score(references, predictions, (tokenizer, model.encoder), 8, layer)
                else:
                    bertscores = bert_score(references, predictions, (tokenizer, model), 8, layer)
            df[f"{model_name}_layer_{layer}"] = bertscores
            df.to_csv(cur_path, index=False) # save dataframe after each layer to avoid data loss

            
if len(sys.argv) == 2:
    data_folder = sys.argv[1] #'yandexgpt' or 'gigachat'
else:
    if len(sys.argv) != 3:
        print("Ошибка. Вы должны ввести название папки с данными ('yandexgpt' или 'gigachat')")
        sys.exit(1)

file_path = os.path.abspath(__file__)

PROJECT_PATH = os.path.dirname(os.path.dirname(file_path))
DATA_PATH = os.path.join(PROJECT_PATH, data_folder)

BERTSCORE_PATH = '/usr/src/app/Dari/bertscore/computed_ru_bertscore'#os.path.join(PROJECT_PATH, 'computed_ru_bertscore')

for folder in os.listdir(DATA_PATH):   
    dataset_path = DATA_PATH+'/'+folder
    if os.path.isdir(dataset_path):
        for csv_file in os.listdir(dataset_path):
            if csv_file.endswith(".csv"):
                print(f'------------------FILE: {csv_file}------------------')
                current_path = BERTSCORE_PATH+'/'+data_folder+'/'+folder
                os.makedirs(current_path, exist_ok=True)
                data_df = pd.read_csv(dataset_path+'/'+csv_file)
                compute_bertscore(data_df, current_path+'/'+csv_file)
