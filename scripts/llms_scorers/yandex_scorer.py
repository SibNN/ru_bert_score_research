import csv
import json
import time

import requests
from tqdm import tqdm

from scripts.paths import YANDEX_SCORES, GIGACHAT_PATH


class YandexScorer:

    def __init__(self, dataset_name: str, score_field: str) -> None:
        self._dataset_dir_path = GIGACHAT_PATH / dataset_name

        self._url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": "Api-Key <ВАШ_API_КЛЮЧ>"
        }

        self._scores_dir_path = YANDEX_SCORES / dataset_name
        self._scores_dir_path.mkdir(parents=True, exist_ok=True)

        self._score_field = score_field
        self._dataset_name = dataset_name

    def run(self):
        for fpath in self._scores_dir_path.iterdir():
            if fpath.suffix != '.csv':
                continue

            all_samples = []

            # Считаем все данные из файла с уже подготовленными первыми n оценками
            with open(fpath, 'r') as fin:
                reader = csv.DictReader(fin)
                fieldnames = list(reader.fieldnames)
                for row in reader:
                    all_samples.append(row)

            with open(fpath, 'w') as fout:
                fieldnames.append(self._score_field)
                writer = csv.DictWriter(fout, fieldnames=fieldnames)
                writer.writeheader()

                for row in tqdm(all_samples, desc=f'Processing {fpath.name}'):

                    input_text = row['output']
                    pred_text = row['pred']
                    prompt = self._get_prompt_3(input_text, pred_text)

                    try:
                        response = requests.post(self._url, headers=self._headers, json=prompt)
                        result = json.loads(response.text)
                    except:
                        print('TimeoutError')
                        result = dict()

                    text = ''

                    try:
                        for m in result['result']['alternatives']:
                            if m['status'] == 'ALTERNATIVE_STATUS_FINAL':
                                text = m['message']['text']
                    except:
                        text = 'FAILED'
                        print(result)

                    row[self._score_field] = text
                    writer.writerow(row)
                    time.sleep(1)

    def _get_prompt_1(self, input_text: str, pred_text: str) -> dict:
        messages = [
            {
                "role": "system",
                "text": "Ты система, которая оценивает качество сгенерированного текста."
            },
            {
                "role": "user",
                "text": f"Оцени качество сгенерированного текста по 5-балльной шкале, "
                        f"насколько он соответствует эталонному тексту."
                        f"Выдай только одно число от 1 до 5, где 1 - сгенерированный текст не соответствует эталонному,"
                        f"5 - сгенерированный текст полностью соответствует эталонному.\n"
                        f"Эталонный текст: {input_text}\n"
                        f"Сгенерированный текст: {pred_text}"
            }
        ]

        prompt = {
            "modelUri": "gpt://<ВАШ_ИДЕНТИФИКАТОР_КАТАЛОГА>/yandexgpt",
            "completionOptions": {
                "stream": False,
                "temperature": 0.6,
                "maxTokens": "2000"
            },
            "messages": messages
        }

        return prompt

    def _get_prompt_2(self, input_text: str, pred_text: str) -> dict:
        dataset2task_desc = {
            'dialogsum_ru': 'Генерация краткого пересказа диалога',
            'reviews_russian': 'Генерация краткого пересказа отзыва об отеле',
            'ru_simple_sent_eval': 'Генерация текста, более простого для понимания',
            'science_summarization_dataset': 'Генерация краткого пересказа научной статьи',
            'telegram-financial-sentiment-summarization': 'Генерация краткого пересказа новостного текста',
            'yandex_jobs': 'Генерация названия должности по описанию вакансии'
        }

        task = dataset2task_desc[self._dataset_name]

        messages = [
            {
                "role": "system",
                "text": "Ты система, которая оценивает качество сгенерированного текста. "
                        f"Задача генерации: {task}."
            },
            {
                "role": "user",
                "text": f"Оцени качество сгенерированного текста по 5-балльной шкале, "
                        f"насколько он соответствует эталонному тексту. "
                        f"Задача генерации: {task}. "
                        f"Выдай только одно число от 1 до 5, где 1 - сгенерированный текст не соответствует эталонному, "
                        f"5 - сгенерированный текст полностью соответствует эталонному.\n"
                        f"Эталонный текст: {input_text}\n"
                        f"Сгенерированный текст: {pred_text}"
            }
        ]

        prompt = {
            "modelUri": "gpt://<ВАШ_ИДЕНТИФИКАТОР_КАТАЛОГА>/yandexgpt",
            "completionOptions": {
                "stream": False,
                "temperature": 0.6,
                "maxTokens": "2000"
            },
            "messages": messages
        }

        return prompt

    def _get_prompt_3(self, input_text: str, pred_text: str) -> dict:
        dataset2task_desc = {
            'dialogsum_ru': 'Генерация краткого пересказа диалога',
            'reviews_russian': 'Генерация краткого пересказа отзыва об отеле',
            'ru_simple_sent_eval': 'Генерация текста, более простого для понимания',
            'science_summarization_dataset': 'Генерация краткого пересказа научной статьи',
            'telegram-financial-sentiment-summarization': 'Генерация краткого пересказа новостного текста',
            'yandex_jobs': 'Генерация названия должности по описанию вакансии'
        }

        task = dataset2task_desc[self._dataset_name]

        messages = [
            {
                "role": "system",
                "text": "Ты система, которая оценивает качество сгенерированного текста. "
                        f"Задача генерации: {task}."
            },
            {
                "role": "user",
                "text": f"Сравни эталонный текст и сгенерированный текст и поставь оценку схожести по смыслу от 1 до 5. "
                        f"Выдай только одно число от 1 до 5, где 1 - сгенерированный текст не соответствует эталонному, "
                        f"5 - сгенерированный текст полностью соответствует эталонному.\n"
                        f"Задача генерации: {task}.\n"
                        f"Эталонный текст: {input_text}\n"
                        f"Сгенерированный текст: {pred_text}"
            }
        ]

        prompt = {
            "modelUri": "gpt://<ВАШ_ИДЕНТИФИКАТОР_КАТАЛОГА>/yandexgpt",
            "completionOptions": {
                "stream": False,
                "temperature": 0.6,
                "maxTokens": "2000"
            },
            "messages": messages
        }

        return prompt


if __name__ == '__main__':
    yandex_scorer = YandexScorer(dataset_name='telegram-financial-sentiment-summarization', score_field='score_3')
    yandex_scorer.run()
