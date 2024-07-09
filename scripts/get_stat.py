import csv

from scripts.paths import DATA_PATH


csv.field_size_limit(10000000)


for dataset_dir_path in DATA_PATH.iterdir():
    if not dataset_dir_path.is_dir():
        continue
    for fpath in dataset_dir_path.iterdir():
        if fpath.suffix != '.csv':
            continue
        n_samples = 0
        n_chars = 0
        unique_samples = set()

        with open(fpath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                unique_samples.add(row['input'])
                n_samples += 1
                n_chars += len(row['input'])

        print(f'{dataset_dir_path.name} / {fpath.name}')
        print(f'N samples: {n_samples}')
        print(f'N chars: {n_chars}')
        print(f'N unique samples: {len(unique_samples)}')
        print('=' * 100)
