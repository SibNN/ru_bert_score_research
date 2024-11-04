# BERTScore для русского языка
В данном репозитории представлено исследование по выбору наиболее релевантных векторных представлений для текстов на русском языке, которые используются в метрике BERTScore. В результате исследования мы сравнили 30 моделей, поддерживающих русский язык. Результаты наших экспериментов показали, что наиболее релевантные векторные представления для русского языка принадлежат 20 слою модели "ai-forever/ru-en-RoSBERTa".

### Авторы:
* Елена Бручес
* Иван Бондаренко
* Дари Батурова

## Обзор
Исследование в оригинальной статье BERTScore охватывало только английский, китайский и турецкий языки, что не дает возможности сделать выводы о том, какая модель лучше всего подходит для оценки качества текстов на русском языке. Поэтому было решено провести анализ корреляции между моделями, поддерживающими русский язык, и экспертными оценками, чтобы разработать рекомендации по выбору оптимальных моделей для оценки качества сгенерированных текстов.

### Наборы данных
Для определения наиболее релевантной модели, с помощью которой оценивается качество сгенерированного текста, мы собрали из открытых источников несколько датасетов, которые различаются по задачам, предметным областям, длине текстов и другим характеристикам. В данной работе мы использовали следующие датасеты:
1. [dialogsum-ru](https://huggingface.co/datasets/d0rj/dialogsum-ru) – датасет для задачи саммаризации диалогов. Исходный датасет DialogSum содержит 13 460 диалогов на английском языке на повседневные темы. Русскоязычный датасет был получен путём перевода текстов на русский язык с помощью Google Translate.
2. [reviews-russian](https://huggingface.co/datasets/trixdade/reviews_russian) – датасет для суммаризации отзывов пользователей на отели и гостиницы. Содержит 92 текста.
3. [ru-simple-sent-eval](https://github.com/dialogue-evaluation/RuSimpleSentEval) – датасет для задачи упрощения текстов на русском языке.
4. [science-summarization-dataset](https://github.com/iis-research-team/summarization-dataset) – датасет для задачи суммаризации научных статей. Содержит 480 статей и их аннотаций из 8 различных научных областей: лингвистика, история, юриспруденция, медицина, компьютерные науки, экономика, химия.
5. [telegram-financial-sentiment-summarization](https://huggingface.co/datasets/mxlcw/telegram-financial-sentiment-summarization) – датасет, который содержит тексты постов из Телеграмма и их краткие содержания. Тексты в данном датасете, в основном, посвящены экономическим и политическим темам.
6. [yandex-jobs](https://huggingface.co/datasets/Kirili4ik/yandex_jobs) – датасет, который содержит описания вакансий Яндекса. Задача заключается в генерации наиболее релевантного названия должности для каждой из вакансий.

|Датасет|Описание|Средняя длина вход. текста|Средняя длина выход. текста|Кол-во уникальных слов во вход. текстах|Кол-во уникальных слов в выход. текстах|
| --- | --- | --- | --- | --- | --- |
|dialogsum-ru|суммаризация диалогов|757|117|9 907|5 400|
|reviews-russian|суммаризация отзывов пользователей на отели и гостиницы|1 390|448|5 096|2 115|
|ru-simple-sent-eval|упрощение текстов|137|91|15 300|20 694|
|science-summarization-dataset|суммаризация научных статей|20 155|843|112 380|13 469|
|telegram-financial-sentiment-summarization|тексты постов из Телеграмма и их краткие содержания|313|169|26 818|20 426|
|yandex-jobs|тексты с описанием вакансий Яндекса|925|38|8 364|504

### Результаты:
#### Корреляции метрик для текстов, сгенерированных Gigachat
|Dataset|Best embedding|BERTScore|Best embedding (ru)|BERTScore (ru)|BLEU|ROUGE-1|ROUGE-2|ROUGE-L|
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|tg|microsoft/mdeberta-v3-base (7)|0.827|ai-forever/ruBert-base (22)|0.825|0.423|0.645|0.497|0.607|
|ru_simple_sent_eval|google/byt5-large (31)|0.615|ai-forever/ru-en-RoSBERTa (20)|0.673|0.096|0.281|0.149|0.25|
|science|facebook/mbart-large-50 (10)|0.749|ai-forever/ru-en-RoSBERTa (20)|0.749|0.282|0.599|0.481|0.560|
|dialogsum_ru|facebook/mbart-large-cc25 (11)|0.447|ai-forever/ru-en-RoSBERTa (7)|0.415|0.158|0.236|0.114|0.247|
|reviews_russian|facebook/mbart-large-50-many-to-many-mmt (6)|0.678|ai-forever/ru-en-RoSBERTa (20)|0.655|0.178|0.346|0.206|0.346|
|yandex|google/byt5-base (6)|0.433|ai-forever/ru-en-RoSBERTa (23)  |0.454|0.078|0.192|0.165|0.192|
|**AVG (слой)**|**google/byt5-large (29)**|**0.594**|**ai-forever/ru-en-RoSBERTa (20)**|**0.630**|**0.191**|**0.379**|**0.257**|**0.359**|
|**AVG (модель)**|**google/byt5-large**|**0.561**|**ai-forever/ruBert-base**|**0.564**|**0.191**|**0.379**|**0.257**|**0.359**|

#### Корреляции метрик для текстов, сгенерированных YandexGPT
|Dataset|Best embedding|BERTScore|Best embedding (ru)|BERTScore (ru)|BLEU|ROUGE-1|ROUGE-2|ROUGE-L|
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|tg|microsoft/mdeberta-v3-base (8)|0.299|ai-forever/ruBert-base (7) |0.345|0.157|0.238|0.197|0.239|
|ru_simple_sent_eval|xlm-roberta-large (16)|0.224|ai-forever/ru-en-RoSBERTa (21) |0.170|0.068|0.079|0.051|0.067|
|science|distilbert-base-multilingual-cased (5)|0.217|ai-forever/ru-en-RoSBERTa (20)|0.199|0.017|0.116|0.134|0.122|
|dialogsum_ru|google/mt5-xl (23)|0.291|ai-forever/ru-en-RoSBERTa (22) |0.333|0.077|0.149|0.157|0.155|
|reviews_russian|google/mt5-xl (23)|0.418|ai-forever/ruSciBERT (10)|0.397|0.138|0.234|0.176|0.213|
|yandex|google/byt5-base (10)|0.509|ai-forever/ruBert-base (0)|0.488|0.169|0.303|0.214|0.301|
|**AVG (слой)**|**google/byt5-base (10)**|**0.284**|**ai-forever/ru-en-RoSBERTa (20)**|**0.275**|**0.096**|**0.171**|**0.138**|**0.166**|
|**AVG (модель)**|**facebook/mbart-large-50-many-to-many-mmt**|**0.249**|**ai-forever/ruBert-base**|**0.260**|**0.096**|**0.171**|**0.139**|**0.166**|

#### Корреляции BERTScore для наиболее релевантных векторных представлений по всем моделям
|Модель|Кол-во параметров|Слой|Pearson|
| --- | --- | --- | --- |
|ai-forever/ru-en-RoSBERTa|404M|20|0.453|
|google/byt5-base|528M|8|0.447|
|google/byt5-large|1.23B|29|0.442|
|facebook/mbart-large-50-many-to-many-mmt|611M|10|0.433|
|facebook/mbart-large-50|611M|10|0.430|
|ai-forever/ruBert-large|427M|22|0.424|
|google/mt5-xl|3.7M|23|0.421|
|microsoft/mdeberta-v3-base|280M|8|0.421|
|ai-forever/ruBert-base|178M|10|0.421|
|google/mt5-large|1.2B|22|0.413|
|facebook/mbart-large-cc25|610M|9|0.410|
|xlm-mlm-100-1280|570M|15|0.403|
|bert-base-multilingual-cased|179M|6|0.401|
|ai-forever/FRED-T5-large|820M|13|0.401|
|ai-forever/ruRoberta-large|355M|20|0.399|
|bond005/rubert-entity-embedder|180M|4|0.398|
|DeepPavlov/rubert-base-cased|180M|7|0.398|
|xlm-roberta-base|279M|6|0.397|
|xlm-roberta-large|561M|16|0.389|
|cointegrated/rubert-tiny2|29М|2|0.385|
|kazzand/ru-longformer-tiny-16384|34.5М|2|0.379|
|distilbert-base-multilingual-cased|135M|3|0.376|
|ai-forever/ruSciBERT|123M|10|0.376|
|kazzand/ru-longformer-large-4096|434М|6|0.373|
|google/mt5-small|300M|3|0.371|
|google/mt5-base|580M|4|0.365|
|ai-forever/ruT5-base|222M|0|0.358|
|google/byt5-small|300M|1|0.341|
|ai-forever/ruT5-large|737M|0|0.341|
|kazzand/ru-longformer-base-4096|148М|6|0.327|

## Подробнее:
Для более подробного изучения нашего исследования вы можете ознакомиться с [презентацией](./RuBERT-Score.pdf).
