from pathlib import Path


file_path = Path(__file__)


PROJECT_PATH = file_path.parent.parent.resolve()
DATA_PATH = PROJECT_PATH / 'data'
RAW_DATA_PATH = PROJECT_PATH / 'raw_data'
YANDEX_PROMPT_PATH = PROJECT_PATH / 'yandexgpt'
GIGACHAT_PATH = PROJECT_PATH / 'gigachat'
