import csv
import json
import os
from pathlib import Path

import pandas as pd


def convert_gazeta(dir_path: str) -> None:
    """ Method to convert .jsonl raw_data to .csv raw_data.
    Suitable for gazeta, russian_xlsum
    """
    for fpath in Path(dir_path).iterdir():
        if fpath.suffix != '.jsonl':
            continue
        samples = list()
        with open(fpath, 'r') as f:
            for line in f:
                sample = json.loads(line)
                samples.append({
                    'text': sample['text'],
                    'summary': sample['summary']
                })

        new_fpath = fpath.with_suffix('.csv')
        with open(new_fpath, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'summary'])
            writer.writeheader()
            writer.writerows(samples)


def convert_parquet_to_csv(fpath: str, rm_parquet_file: bool = True) -> None:
    new_fpath = fpath.replace('.parquet', '.csv')
    df = pd.read_parquet(fpath)
    df.to_csv(new_fpath, index=False)

    if rm_parquet_file:
        os.remove(fpath)


def convert_parquet_to_csv_dir(dir_path: str) -> None:
    for fpath in Path(dir_path).iterdir():
        convert_parquet_to_csv(str(fpath))


def convert_ru_ego_literature(fpath: str) -> None:
    new_fpath = fpath.replace('.csv', '_cleaned.csv')
    df = pd.read_csv(fpath, delimiter='\t')
    new_df = df[['chapter', 'brief_chapter']]
    new_df.to_csv(new_fpath, index=False)


def convert_curation_corpus_ru(fpath: str, rm_parquet_file: bool = True) -> None:
    convert_parquet_to_csv(fpath)

    new_fpath = fpath.replace('.parquet', '_cleaned.csv')
    csv_fpath = fpath.replace('.parquet', '.csv')

    df = pd.read_csv(csv_fpath)
    new_df = df[['summary', 'article_content']]

    new_df.to_csv(new_fpath, index=False)

    if rm_parquet_file:
        os.remove(fpath)


def convert_tg_fin_sentiment_analysis(fpath: str, rm_initial_file: bool = True) -> None:
    new_fpath = fpath.replace('.csv', '_cleaned.csv')
    df = pd.read_csv(fpath)

    new_df = df[['text', 'summarized_text']]
    new_df.to_csv(new_fpath)

    if rm_initial_file:
        os.remove(fpath)
