import os
import gc

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig
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


def compute_bertscore(df: pd.DataFrame, cur_path: str) -> None:
    model_names = ['bert-base-multilingual-cased',
                       'xlm-mlm-100-1280', 
                       'distilbert-base-multilingual-cased', 
                       'xlm-roberta-base', 'xlm-roberta-large', 
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
    for model in model_names:
        print(f'MODEL: {model}')
        num_layers = get_n_layers(model)
        for layer in tqdm(range(num_layers)):
            bertscores = []
            for index, row in df.iterrows():
                pred_text = row['pred']
                ref_text = row['output']
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
            df.to_csv(cur_path, index=False)


file_path = os.path.abspath(__file__)

PROJECT_PATH = os.path.dirname(os.path.dirname(file_path))
GIGACHAT_PATH = os.path.join(PROJECT_PATH, 'gigachat')
BERTSCORE_PATH = os.path.join(PROJECT_PATH, 'computed_bertscore')

bertscore = load("bertscore")

for folder in os.listdir(GIGACHAT_PATH):
    data_path = GIGACHAT_PATH+'/'+folder
    if os.path.isdir(data_path):
        for csv_file in os.listdir(data_path):
            if csv_file.endswith(".csv"):
                print(f'------------------FILE: {csv_file}------------------')
                os.makedirs(BERTSCORE_PATH+'/'+folder, exist_ok=True)
                gigachat_df = pd.read_csv(data_path+'/'+csv_file)
                bertscore_dict = compute_bertscore(gigachat_df, BERTSCORE_PATH+'/'+folder+'/'+csv_file)
                