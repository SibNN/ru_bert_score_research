# Описание файлов
## Генерация данных
### Gigachat
Запустите `gigachat_generator.py` для генерации ответов с помощью модели Gigachat:
```sh
python scripts/gigachat_generator.py
```
Для датасета 'science_summarization_dataset' необходимо запустить отдельный скрипт:
```sh
python scripts/gigachat_generator_sсience.py
```
В директории `/gigachat/DATASET_NAME/` нужно создать файл `prompt.txt` с соответствующим промптом для данной задачи.

DATASET_NAME - название датасета ('dialogsum_ru', 'reviews_russian','ru_simple_sent_eval', 'science_summarization_dataset', 'telegram-financial-sentiment-summarization')

Вставьте название необходимого датасета в код на 82 строке
```python
gigachat = GigachatGenerator(DATASET_NAME)
```
### YandexGPT
Запустите `yandex_gpt_generator.py` для генерации ответов с помощью модели YandexGPT:
```sh
python scripts/yandex_gpt_generator.py
```
Для датасета 'science_summarization_dataset' необходимо запустить отдельный скрипт:
```sh
python scripts/yandex_gpt_generator_science.py
```
В директории `/yandexgpt/DATASET_NAME/` нужно создать файл `messages.json` с соответствующим промптом для данной задачи.

DATASET_NAME - название датасета ('dialogsum_ru', 'reviews_russian','ru_simple_sent_eval', 'science_summarization_dataset', 'telegram-financial-sentiment-summarization')

Вставьте название необходимого датасета в код на 98 строке
```python
gen = YandexGPTGenerator(DATASET_NAME)
```
### Фильтрация предсказаний от YandexGPT
Запустите `filter_yandexgpt.py` для очистки от некорректных предсказаний:
```sh
python scripts/filter_yandexgpt.py
```
## Получение экспертной оценки для сгенерированных ответов
### Gigachat
Запустите `compare_gigachat.py` для получения экспертной оценки от модели Gigachat для всех датасетов:
```sh
python scripts/compare_gigachat.py
```
После выполнения скрипта в тех же директориях, где находились исходные CSV-файлы, появятся обновленные файлы с 3 добавленными колонками для оценок, где будут храниться результаты оценки от Gigachat.
### YandexGPT
Запустите `yandex_scorer.py` для получения экспертной оценки от модели YandexGPT:
```sh
python scripts/yandex_scorer.py
```
Вставьте название необходимого датасета и название колонки с оценками (например, 'score_1') в код на 188 строке
```python
yandex_scorer = YandexScorer(dataset_name=DATASET_NAME, score_field=COLUMN_NAME)
```
После выполнения скрипта в тех же директориях, где находились исходные CSV-файлы, появятся обновленные файлы с добавленной колонкой для оценок, где будут храниться результаты оценки от Yandex API.
## Подсчет BERTScore с помощью различных моделей
Запустите `compute_bertscore.py` со следующими параметрами:

DATA_FOLDER - название папки с данными ('yandexgpt' или 'gigachat')

MODEL_LANG - языковой тип моделей ('multilingual' или 'ru')
```sh
python scripts/compute_bertscore.py DATA_FOLDER MODEL_LANG
```
Результаты мультиязычных моделей будут сохранены в папку `computed_bertscore`, а русскоязычных моделей - в папку `computed_ru_bertscore`.

У некоторых моделей помимо колонок со значениями BERTScore, полученных с каждого слоя, есть колонки 'pred_truncated_MODEL_NAME' и 'ref_truncated_MODEL_NAME' c булевыми значениями (True - предсказанный/эталонный текст был обрезан, False - не был обрезан).
## Расчет традиционных метрик 
Запустите `compute_stat_metrics.py` для получения значений метрик BLEU и ROUGE:
```sh
python scripts/compute_stat_metrics.py
```
Вставьте путь до необходимых данных в код на 9 и 98 строках
```python
from scripts.paths import DATA_PATH
```
```python
metrics_computer = StatMetricsComputer(DATA_PATH)
```
Вместо DATA_PATH вставьте 'GIGACHAT_SCORE' или 'YANDEX_SCORES'.

После выполнения скрипта в тех директориях, куда направляет путь DATA_PATH, исходные файлы дополнятся добавленными колонками с традиционными метриками.
## Расчет корреляций между эталонными оценками и BERTScore
Запустите `compute_corr.py` со следующими параметрами:

DATA_FOLDER - название папки с данными ('yandexgpt' или 'gigachat')

MODEL_LANG - языковой тип моделей ('multilingual' или 'ru')
```sh
python scripts/compute_corr.py DATA_FOLDER MODEL_LANG
```
Результаты для мультиязычных моделей будут сохранены в папку `correlations`, а для русскоязычных моделей - в папку `ru_correlations`.