# TAP-RSV

## 1. Get started
unzip the data.zip and put into `./data/` folder

## 2. Generally, to run each model
```bash
python train_arima.py --val_start "2022-07-23" --test_start "2022-08-20"
python train_lstm.py --val_start "2022-07-23" --test_start "2022-08-20"
python train_transformer.py --val_start "2022-07-23" --test_start "2022-08-20"
python train_xgboost.py --val_start "2022-07-23" --test_start "2022-08-20"
python train_stan.py --val_start "2022-07-23" --test_start "2022-08-20"
python train_hoist.py --val_start "2022-07-23" --test_start "2022-08-20"
python train_taprsv.py --val_start "2022-07-23" --test_start "2022-08-20"
```

## Analysis 1: RSV disease cluster
> please use `./notebook/Task 1 - IQVIA disease cluster.ipynb`

## Analysis 2: RSV location cluster
> please use `./notebook/Task 2 - IQVIA location cluster.ipynb`

> please use `./notebook/Task 2 - CDC location cluster.ipynb`

> please use `./notebook/Task 2 - Google location cluster.ipynb`

- for humidity and temperature correlation, please  use `weather.py` to crawl the climate data. 
> please use `./notebook/Task 2 - humidity and temperature correlation.ipynb`

## Prediction - Result 1: RSV 2022-2023 season prediction
- run `parallel1-exp1.sh` and `parallel2-exp1.sh`. They can be run in parallel. The results are already saved into `./logs/results-exp1` and the tensorboard logs are saved into `./logs/RSV-analysis-exp1`.
- Each file in `./logs/results-exp1` is of 40 lines (every 2 lines are the same, just ignore that). So, there are only 20 meaningful lines in each file. 
    - Line 1-10 represents the results of 5 runs with different random seeds over prediction window 08/20/2022 -> 09/17/2022
    - Line 11-20 represents the results of 5 runs with different random seeds over prediction window 09/17/2022 -> 10/15/2022
    - Line 21-30 represents the results of 5 runs with different random seeds over prediction window 10/15/2022 -> 11/12/2022
    - Line 31-40 represents the results of 5 runs with different random seeds over prediction window 11/12/2022 -> 12/10/2022
- You can use `./src/result-extraction-exp1.py` to extract the results from the files in `./logs/results-exp1` and then use `https://tableconvert.com/csv-to-csv` to convert the csv format into table format. The table format can be directly copied to the word document.

## Prediction - Result 2: map
- use the `./boken_map/draw_IQVIA.py` to generate the true and predicted map for 10/15/2022 -> 11/12/2022.
- click to see the map https://ycq091044.github.io/TAP-RSV/

## Supp 1
- run `parallel1-exp2.sh` and `parallel2-exp2.sh`. They can be run in parallel. The results are already saved into `./logs/results-exp2` and the tensorboard logs are saved into `./logs/RSV-analysis-exp2`.
- The format of the data in `./logs/results-exp2` is the same as `./logs/results-exp1`.
- use `./notebook/Exp 2 - longer-window.ipynb` to generate the results.

## Supp 2
- you will have to modify the inner structure of TAP-RSV model and run the variant one by one. The results are already saved into `./logs/results-exp3` and the tensorboard logs are saved into `./logs/RSV-analysis-exp3`.
- use `./notebook/Exp 3 - feature-ablation.ipynb` to generate the results.