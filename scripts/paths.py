from pathlib import Path


file_path = Path(__file__)


PROJECT_PATH = file_path.parent.parent.resolve()
DATA_PATH = PROJECT_PATH / 'data'
RAW_DATA_PATH = PROJECT_PATH / 'raw_data'
YANDEX_PROMPT_PATH = PROJECT_PATH / 'yandexgpt'
GIGACHAT_PATH = PROJECT_PATH / 'gigachat'
BERTSCORE_PATH = PROJECT_PATH / 'computed_bertscore'
YANDEX_SCORES = PROJECT_PATH / 'yandex_scores'
GIGACHAT_SCORE = PROJECT_PATH / 'gigachat_score'
CONFIG_PATH = PROJECT_PATH / 'scripts' / 'config.yml'
