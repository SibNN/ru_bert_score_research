import sys
import csv
import json
import time
from collections import defaultdict
from copy import deepcopy
from typing import Optional, List

import requests
from tqdm import tqdm
import yaml

from scripts.paths import YANDEX_PROMPT_PATH, DATA_PATH, CONFIG_PATH


config = yaml.safe_load(open(CONFIG_PATH))


class YandexGPTGenerator:

    CHARS_LIMIT = 20000  # YandexGPT input is limited by 8 192 tokens

    def __init__(self, dataset_name: str) -> None:
        self._url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {config['api_key']}"
        }

        self._dataset_name = dataset_name

        self._dataset_path = DATA_PATH / dataset_name
        self._model_artifacts_path = YANDEX_PROMPT_PATH / dataset_name

        prompt_path = self._model_artifacts_path / 'messages.json'
        with open(prompt_path, 'r') as f:
            self._messages = json.load(f)

    def run(self, fnames_to_run: Optional[List[str]] = None) -> None:
        for fpath in self._dataset_path.iterdir():

            if fnames_to_run and fpath.name not in fnames_to_run:
                continue

            output_fpath = self._model_artifacts_path / fpath.name

            input_output_texts = defaultdict(list)

            with open(fpath, 'r') as fin, open(output_fpath, 'w') as fout:
                reader = csv.DictReader(fin)
                for row in reader:
                    input_output_texts[row['input']].append(row['output'])

                writer = csv.DictWriter(fout, fieldnames=['input', 'output', 'pred'])
                writer.writeheader()
                for input_text in tqdm(input_output_texts, desc=f'Processing {fpath.name}'):
                    prompt = self._get_prompt(input_text[:self.CHARS_LIMIT])
                    response = requests.post(self._url, headers=self._headers, json=prompt)
                    result = json.loads(response.text)

                    text = ''

                    try:
                        for m in result['result']['alternatives']:
                            if m['status'] == 'ALTERNATIVE_STATUS_FINAL':
                                text = m['message']['text']
                    except:
                        text = 'FAILED'
                        print(result)

                    outputs = input_output_texts[input_text]
                    for output in outputs:
                        row = {
                            'input': input_text,
                            'output': output,
                            'pred': text
                        }
                        writer.writerow(row)

                    time.sleep(1)

    def _get_prompt(self, input_text: str) -> dict:
        messages = deepcopy(self._messages)
        for d in messages:
            if d['role'] == 'user':
                d['text'] += input_text
                break

        prompt = {
            "modelUri": f"gpt://{config['catalog_id']}/yandexgpt",
            "completionOptions": {
                "stream": False,
                "temperature": 0.6,
                "maxTokens": "2000"
            },
            "messages": messages
        }

        return prompt


if __name__ == '__main__':
    if len(sys.argv) == 2:
        dataset_name = sys.argv[1] #'dialogsum_ru', 'reviews_russian','ru_simple_sent_eval', 'telegram-financial-sentiment-summarization'
    else:
        raise ValueError("Неправильное количество аргументов. Ожидался 1 аргумент: название датасета ('dialogsum_ru', 'reviews_russian','ru_simple_sent_eval', 'telegram-financial-sentiment-summarization').")

    gen = YandexGPTGenerator(dataset_name)
    gen.run()
