import sys
import csv
from pathlib import Path
from typing import Dict

from bleu import list_bleu
from rouge_metric import PyRouge
from tqdm import tqdm

from scripts.paths import GIGACHAT_SCORE, YANDEX_SCORES


class StatMetricsComputer:

    def __init__(self, data_dir: str) -> None:
        self._data_dir = Path(data_dir)

    def run(self):
        for dataset_path in self._data_dir.iterdir():
            for fpath in dataset_path.iterdir():
                if fpath.suffix != '.csv':
                    continue
                rows = []
                with open(fpath, 'r') as fin:
                    reader = csv.DictReader(fin)
                    fieldnames = list(reader.fieldnames)
                    fieldnames.extend(['bleu', 'rouge-1', 'rouge-2', 'rouge-l'])
                    for row in reader:
                        rows.append(row)
                with open(fpath, 'w') as fout:
                    writer = csv.DictWriter(fout, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in tqdm(rows, f'Scores for {fpath.name}'):
                        try:
                            bleu_value = self._count_bleu(row['output'], row['pred'])
                        except:
                            bleu_value = -1
                        row['bleu'] = bleu_value

                        rouge_values = self._count_rouge(row['output'], row['pred'])
                        row.update(rouge_values)

                        writer.writerow(row)

    def _count_bleu(self, ref_text: str, hyp_text: str) -> float:
        bleu_value = list_bleu([ref_text], [hyp_text])
        return bleu_value

    def _count_rouge(self, ref_text: str, hyp_text: str) -> Dict[str, float]:
        rouge = PyRouge(rouge_n=(1, 2), rouge_l=True, rouge_w=False,
                        rouge_w_weight=1.2, rouge_s=False, rouge_su=False, skip_gap=4)
        scores = rouge.evaluate([hyp_text], [[ref_text]])

        rouge_metrics = dict()

        # extract only f-score for every rouge type
        for score in scores:
            rouge_metrics[score] = round(scores[score]['f'], 2)

        return rouge_metrics


if __name__ == '__main__':
    if len(sys.argv) == 2:
        score_model = sys.argv[1] #'yandexgpt' или 'gigachat'
    else:
        raise ValueError("Неправильное количество аргументов. Ожидался 1 аргумент: название модели, от которой были получены оценки ('yandexgpt' или 'gigachat').")
    if score_model=='yandexgpt':
        data_folder_name = YANDEX_SCORES
    elif score_model=='gigachat':
        data_folder_name = GIGACHAT_SCORE
    metrics_computer = StatMetricsComputer(data_folder_name)
    metrics_computer.run()
