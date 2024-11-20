import sys
import csv
from pathlib import Path
from typing import Optional, List

from gigachat import GigaChat
from tqdm import tqdm
import yaml

from scripts.paths import GIGACHAT_PATH, DATA_PATH, CONFIG_PATH


config = yaml.safe_load(open(CONFIG_PATH))


class GigachatGenerator:
    """
    Класс для генерации текстов с помощью Gigachat.

    В директории /gigachat/dataset_name/ нужно создать файл prompt.txt с соответствующим
    промптом для данной задачи.

    Args:
        dataset_name: название датасета, на котором нужно запустить модель.
    """

    CREDS = config['credentials']

    def __init__(self, dataset_name: str) -> None:
        self._dataset_name = dataset_name

        self._dataset_path = DATA_PATH / dataset_name
        self._model_artifacts_path = GIGACHAT_PATH / dataset_name

        self._prompt = config['gigachat'][dataset_name].strip()

    def run(self, fnames_to_run: Optional[List[str]] = None) -> None:
        """
        Запуск модель на указанном датасете.

        :param fnames_to_run: Список строковых названий файлов (не полных путей), на которых
            нужно запустить модель. Если список не передан, то модель будет запущена на всём датасете.
        :return: None
        """
        for i, fpath in enumerate(self._dataset_path.iterdir()):
            if fnames_to_run and fpath.name not in fnames_to_run:
                continue
            output_fpath = self._model_artifacts_path / fpath.name

            self.run_on_file(fpath, output_fpath)

    def run_on_file(self, input_path: Path, output_path: Path) -> None:
        """
        Запуск модели на одном файле.

        :param input_path: Путь до исходного файла
        :param output_path: Путь до файла, куда будут записаны предсказания
        :return: None
        """

        unique_inputs = set()

        with open(input_path, 'r') as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                unique_inputs.add(row['input'])

        with open(output_path, 'w') as fout:
            writer = csv.DictWriter(fout, fieldnames=['input', 'pred'])
            writer.writeheader()
            for input_text in tqdm(unique_inputs, desc=f'Processing {input_path.name}'):
                prompt = f'{self._prompt} {input_text}'

                with GigaChat(credentials=self.CREDS, verify_ssl_certs=False) as giga:
                    response = giga.chat(prompt)
                    text = response.choices[0].message.content

                    writer.writerow({
                        'input': input_text,
                        'pred': text
                    })


if __name__ == '__main__':
    if len(sys.argv) == 2:
        dataset_name = sys.argv[1] #'dialogsum_ru', 'reviews_russian','ru_simple_sent_eval', 'telegram-financial-sentiment-summarization').
    else:
        raise ValueError("Неправильное количество аргументов. Ожидался 1 аргумент: название датасета ('dialogsum_ru', 'reviews_russian','ru_simple_sent_eval', 'telegram-financial-sentiment-summarization').")

    gigachat = GigachatGenerator(dataset_name)
    gigachat.run()
