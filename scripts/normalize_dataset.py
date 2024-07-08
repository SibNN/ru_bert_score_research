import csv

from scripts.paths import RAW_DATA_PATH, DATA_PATH


csv.field_size_limit(100000000)


class DatasetNormalizer:
    """ Класс для нормализации датасетов """

    INPUT = 'input'
    OUTPUT = 'output'

    def run(self, dataset_name: str, input_name: str, output_name: str) -> None:
        """ Запускает нормализацию указанного датасета, который лежит в директории raw_data/.

        :param dataset_name: имя датасета
        :param input_name: имя поля, которое будет использоваться для входа модели
        :param output_name: имя поля с "золотым" сгенерированным текстом.
        :return:
        """
        raw_dataset_dir = RAW_DATA_PATH / dataset_name
        dataset_dir = DATA_PATH / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for raw_fpath in raw_dataset_dir.iterdir():
            if raw_fpath.suffix != '.csv':
                continue
            fpath = dataset_dir / raw_fpath.name
            with open(raw_fpath, 'r') as fin, open(fpath, 'w') as fout:
                reader = csv.DictReader(fin)
                writer = csv.DictWriter(fout, fieldnames=[self.INPUT, self.OUTPUT])
                writer.writeheader()

                for row in reader:
                    writer.writerow({
                        self.INPUT: row[input_name],
                        self.OUTPUT: row[output_name]
                    })
