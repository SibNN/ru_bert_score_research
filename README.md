# ru-bert-score

Link to normalized datasets: https://disk.yandex.ru/d/VjTEGAhKaEDRBw

Download them to the '/data' dir.

## Stat for the datasets (processed)

| **№** |                 **Dataset**                |                      **File**                     | **N samples** | **N unique samples** | **N chars** | **Gigachat** | **YandexGPT** |
|:-----:|:------------------------------------------:|:-------------------------------------------------:|:-------------:|:--------------------:|:-----------:|:------------:|:-------------:|
|   1   | telegram-financial-sentiment-summarization | telegram_data_cleaned.csv                         | 18 107        |        18 107        | 10 407 423  |    6 229     |               |
|   2   | ru_simple_sent_eval                        | dev_sents.csv                                     | 3 406         |        1 000         |   470 025   |      +       |               |
|   3   | ru_simple_sent_eval                        | public_test_sents.csv                             | 3 398         |        1 000         |   461 214   |      +       |               |
|   4   | matreshka                                  | train-00000-of-00001-287def2e1da553e7.csv         | 6 655         |        6 646         |  2 167 490  |              |               |
|   5   | gazeta                                     | gazeta_test.csv                                   | 6 793         |        6 793         | 30 202 257  |              |               |
|   6   | gazeta                                     | gazeta_val.csv                                    | 6 369         |        6 369         | 27 758 680  |              |               |
|   7   | gazeta                                     | gazeta_train.csv                                  | 60 964        |        60 844        | 275 878 489 |              |               |
|   8   | russian_xlsum                              | russian_val.csv                                   | 7 780         |        7 779         | 26 165 974  |              |               |
|   9   | russian_xlsum                              | russian_test.csv                                  | 7 780         |        7 780         | 26 100 838  |              |               |
|   10  | russian_xlsum                              | russian_train.csv                                 | 62 243        |        62 211        | 252 415 678 |              |               |
|   11  | russe_detox_2022                           | dev.csv                                           | 800           |         800          |   50 907    |              |               |
|   12  | russe_detox_2022                           | train.csv                                         | 6 948         |        6 948         |   444 692   |              |               |
|   13  | dialogsum_ru                               | test-00000-of-00001-2f13615b955ea947.csv          | 1 500         |         499          |  1 135 638  |      +       |               |
|   14  | dialogsum_ru                               | train-00000-of-00001-bcc43b46acda4001.csv         | 12 460        |        11 598        |  9 188 691  |              |               |
|   15  | dialogsum_ru                               | validation-00000-of-00001-7e263d81db1c7a12.csv    | 500           |         498          |   363 179   |              |               |
|   16  | ru_dial_sum                                | head3111.csv                                      | 3 111         |        3 111         | 12 948 611  |              |               |
|   17  | ru_dial_sum                                | tail300.csv                                       | 300           |         300          |  1 095 179  |              |               |
|   18  | reviews_russian                            | train.csv                                         | 95            |          93          |   131 248   |      +       |               |
|   19  | reviews_russian                            | test.csv                                          | 15            |          15          |   28 248    |      +       |               |
|   20  | yandex_jobs                                | vacancies.csv                                     | 625           |         528          |   578 066   |              |               |
|   21  | ru-ego-literature                          | ru-ego-literature_cleaned.csv                     | 532           |         532          | 10 248 279  |              |               |
|   22  | samsum_ru                                  | test-00000-of-00001-e5495f81e41a1924.csv          | 819           |         819          |   428 689   |              |               |
|   23  | samsum_ru                                  | train-00000-of-00001-76cc3fe8132d8f4b.csv         | 14 731        |        14 248        |  7 569 675  |              |               |
|   24  | samsum_ru                                  | validation-00000-of-00001-111fbc2081be60b7.csv    | 818           |         818          |   409 594   |              |               |
|   25  | curation_corpus_ru                         | train-00000-of-00001-1d043be9cad23f73_cleaned.csv | 30 454        |        30 454        | 112 924 220 |              |               |