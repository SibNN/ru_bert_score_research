import csv
import json

import requests
from tqdm import tqdm

from scripts.paths import YANDEX_PROMPT_PATH, DATA_PATH


class YandexGPTGenerator:

    def __init__(self, dataset_name: str) -> None:
        self._url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": "Api-Key AQVN1ymnksB0EAKoby7H1X07zvYXBiLeHbtrtGcV"
        }

        self._dataset_name = dataset_name

        self._dataset_path = DATA_PATH / dataset_name
        self._model_artifacts_path = YANDEX_PROMPT_PATH / dataset_name

        prompt_path = self._model_artifacts_path / 'messages.json'
        with open(prompt_path, 'r') as f:
            self._messages = json.load(f)

    def run(self):
        for fpath in self._dataset_path.iterdir():
            output_fpath = self._model_artifacts_path / fpath.name
            with open(fpath, 'r') as fin, open(output_fpath, 'w') as fout:
                reader = csv.DictReader(fin)
                writer = csv.DictWriter(fout, fieldnames=['input', 'output', 'pred'])
                writer.writeheader()
                for row in tqdm(reader, desc=f'Processing {fpath.name}'):
                    input_text = row['input']
                    prompt = self._get_prompt(input_text)
                    response = requests.post(self._url, headers=self._headers, json=prompt)
                    result = json.loads(response.text)

                    text = ''

                    try:
                        for m in result['result']['alternatives']:
                            if m['status'] == 'ALTERNATIVE_STATUS_FINAL':
                                text = m['message']['text']
                    except:
                        text = 'FAILED'

                    new_row = row.copy()
                    new_row['pred'] = text

                    writer.writerow(new_row)

    def _get_prompt(self, input_text: str) -> dict:
        messages = self._messages.copy()
        for d in messages:
            if d['role'] == 'user':
                d['text'] += input_text
                break

        prompt = {
            "modelUri": "gpt://b1g6gvssn1feurraaho0/yandexgpt",
            "completionOptions": {
                "stream": False,
                "temperature": 0.6,
                "maxTokens": "2000"
            },
            "messages": messages
        }

        return prompt


if __name__ == '__main__':
    gen = YandexGPTGenerator('dialogsum_ru')
    gen.run()



# идентификатор ключа: ajen56jp5g3k201agrpe
# секретный ключ: AQVN1ymnksB0EAKoby7H1X07zvYXBiLeHbtrtGcV
