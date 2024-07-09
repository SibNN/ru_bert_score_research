from gigachat import GigaChat

import csv

from tqdm import tqdm

from scripts.paths import GIGACHAT_PATH, DATA_PATH


class GigachatGenerator:

    CREDS = 'M2Y4MGIwZmItOGY0NS00YjE2LTgzZTItZjE2OWE0YzJiMWMwOmM4MTlhNmUxLTFhNWQtNDQ0Ny1iYzRmLTZkOGIzOTQwNjIzYQ=='

    def __init__(self, dataset_name: str) -> None:
        self._dataset_name = dataset_name

        self._dataset_path = DATA_PATH / dataset_name
        self._model_artifacts_path = GIGACHAT_PATH / dataset_name

        prompt_path = self._model_artifacts_path / 'prompt.txt'
        with open(prompt_path, 'r') as f:
            self._prompt = f.read().strip()

    def run(self):
        for fpath in self._dataset_path.iterdir():
            output_fpath = self._model_artifacts_path / fpath.name
            with open(fpath, 'r') as fin, open(output_fpath, 'w') as fout:
                reader = csv.DictReader(fin)
                writer = csv.DictWriter(fout, fieldnames=['input', 'output', 'pred'])
                writer.writeheader()
                for row in tqdm(reader, desc=f'Processing {fpath.name}'):
                    input_text = row['input']
                    prompt = f'{self._prompt} {input_text}'

                    with GigaChat(credentials=self.CREDS, verify_ssl_certs=False) as giga:
                        response = giga.chat(prompt)
                        text = response.choices[0].message.content

                    new_row = row.copy()
                    new_row['pred'] = text

                    writer.writerow(new_row)
            break


if __name__ == '__main__':
    gigachat = GigachatGenerator('dialogsum_ru')
    gigachat.run()
