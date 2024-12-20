import csv
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional, List

import requests
from tqdm import tqdm
import yaml

from scripts.paths import YANDEX_PROMPT_PATH, CONFIG_PATH


config = yaml.safe_load(open(CONFIG_PATH))


class YandexGPTGenerator:

    CHARS_LIMIT = 20000  # YandexGPT input is limited by 8 192 tokens

    def __init__(self, dataset_path: str) -> None:
        self._url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Api-Key {config['api_key']}"
        }

        self._dataset_path = Path(dataset_path)
        self._model_artifacts_path = YANDEX_PROMPT_PATH / 'science_summarization_dataset'

        self._messages = config['yandexgpt']['science_summarization_dataset']

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
    dataset_path = config['science_dataset_path']
    gen = YandexGPTGenerator(dataset_path)
    gen.run()
