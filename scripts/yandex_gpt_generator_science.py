import csv
import json
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Optional, List

import requests
from tqdm import tqdm

from scripts.paths import YANDEX_PROMPT_PATH


class YandexGPTGenerator:

    CHARS_LIMIT = 20000  # YandexGPT input is limited by 8 192 tokens

    def __init__(self, dataset_path: str) -> None:
        self._url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": "Api-Key AQVN1mlx5gNXrfw6a6tOZ9Oj2ZtV3n7IvGSPT5R4"
        }

        self._dataset_path = Path(dataset_path)
        self._model_artifacts_path = YANDEX_PROMPT_PATH / 'science_summarization_dataset'

        prompt_path = self._model_artifacts_path / 'messages.json'
        with open(prompt_path, 'r') as f:
            self._messages = json.load(f)

    def run(self, fnames_to_run: Optional[List[str]] = None) -> None:
        with open(self._model_artifacts_path / 'preds.csv', 'w') as fout:
            writer = csv.DictWriter(fout, fieldnames=['dir_name', 'input', 'output', 'pred'])
            writer.writeheader()

            for science_dir in self._dataset_path.iterdir():
                # идём по директориям с доменами: chemistry, economics, etc.
                if not science_dir.is_dir():
                    continue
                for paper_dir in tqdm(science_dir.iterdir(), desc=f'Processing {science_dir.name}'):
                    if not paper_dir.is_dir():
                        continue

                    with open(paper_dir / 'abstract.txt', 'r') as f:
                        abstract = f.read().strip()
                    with open(paper_dir / 'text.txt', 'r') as f:
                        text = f.read().strip()
                        prompt = self._get_prompt(text[:self.CHARS_LIMIT])
                        response = requests.post(self._url, headers=self._headers, json=prompt)
                        result = json.loads(response.text)

                        pred = ''

                        try:
                            for m in result['result']['alternatives']:
                                if m['status'] == 'ALTERNATIVE_STATUS_FINAL':
                                    pred = m['message']['text']
                        except:
                            pred = 'FAILED'
                            print(result)

                        writer.writerow({
                            'dir_name': paper_dir.name,
                            'input': text,
                            'output': abstract,
                            'pred': pred
                        })

                        time.sleep(1)

    def _get_prompt(self, input_text: str) -> dict:
        messages = deepcopy(self._messages)
        for d in messages:
            if d['role'] == 'user':
                d['text'] += input_text
                break

        prompt = {
            "modelUri": "gpt://b1gk85hjrhd3k9deoh0s/yandexgpt",
            "completionOptions": {
                "stream": False,
                "temperature": 0.6,
                "maxTokens": "2000"
            },
            "messages": messages
        }

        return prompt


if __name__ == '__main__':
    dataset_path = '/Users/elena/Documents/summarization/summarization-dataset/dataset/'
    gen = YandexGPTGenerator(dataset_path)
    gen.run()



# идентификатор ключа: ajen56jp5g3k201agrpe
# секретный ключ: AQVN1ymnksB0EAKoby7H1X07zvYXBiLeHbtrtGcV

# sibnn:

# Идентификатор ключа:
# ajemd5ouc0qnlnu3kqtb
# Ваш секретный ключ:
# AQVN1mlx5gNXrfw6a6tOZ9Oj2ZtV3n7IvGSPT5R4

