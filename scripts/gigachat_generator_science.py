from pathlib import Path

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

    def __init__(self, dataset_path: str) -> None:

        self._dataset_path = DATA_PATH / dataset_path
        self._model_artifacts_path = GIGACHAT_PATH / 'science_summarization_dataset'

        self._prompt = config['gigachat']['science_summarization_dataset'].strip()

    def run(self) -> None:
        """
        Запуск модель на указанном датасете.

        :return: None
        """
        for science_dir in self._dataset_path.iterdir():
            # идём по директориям с доменами: chemistry, economics, etc.
            if not science_dir.is_dir():
                continue
            for paper_dir in tqdm(science_dir.iterdir(), desc=f'Processing {science_dir.name}'):
                if not paper_dir.is_dir():
                    continue
                # идём по директориям со статьями: inf_1, inf_2, inf_3, etc.
                output_dir_path = self._model_artifacts_path / science_dir.name / paper_dir.name
                output_dir_path.mkdir(parents=True, exist_ok=True)

                self.run_on_file(paper_dir / 'text.txt', output_dir_path / 'pred.txt')

    def run_on_file(self, input_path: Path, output_path: Path) -> None:
        """
        Запуск модели на одном файле.

        :param input_path: Путь до исходного файла
        :param output_path: Путь до файла, куда будут записаны предсказания
        :return: None
        """

        with open(input_path, 'r') as f:
            input_text = f.readlines()
            prompt = f'{self._prompt} {input_text}'

        with GigaChat(credentials=self.CREDS, verify_ssl_certs=False) as giga:
            response = giga.chat(prompt)
            text = response.choices[0].message.content

        with open(output_path, 'w') as f:
            f.write(text)


if __name__ == '__main__':
    dataset_path = config['science_dataset_path']
    gigachat = GigachatGenerator(dataset_path)
    gigachat.run()
