import pickle
import os
import argparse

import numpy as np
import torch
import  torch.nn as nn

from utils import load_dataset, mean_absolute_percentage_error, ccc, r_squared, pcc
from pmdarima import auto_arima
from tqdm import tqdm
   

def prepare_dataloader(args):
    # set random seed
    # seed = 4523
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    
    # load data
    """
    county (2334, 14)
    covid (2334, 639)
    distance (2334, 2334)
    mob (2334, 2334)
    vac (2334, 639, 34)
    hos (2334, 639, 4)
    claim (2334, 639, 20)
    bron (2334, 264, 19)
    weeks (264,)
    """
    county, covid, distance, mob, vac, hos, claim, bron, weeks, fips, disease_tensor_emb, county_tensor_emb = load_dataset()
    
    # "bron" is the disease tensor, already in week level
    bron = np.array(bron, dtype=np.float32) #/ county[:, 1].reshape(-1, 1, 1)

    # county normalization
    county = (county - county.mean(axis=0)) / (county.std(axis=0) + 1e-8)
    county = np.array(county, dtype=np.float32)
    
    # covid normalization -> weekly (2334, 91)
    covid = np.diff(np.concatenate([np.zeros((covid.shape[0], 1)), np.cumsum(covid, 1)], 1)[:, ::7], axis=1)
    covid = (covid - covid.mean(axis=0)) / (covid.std(axis=0) + 1e-8)
    covid = np.array(covid, dtype=np.float32)
    
    # vac normalization -> weekly (2334, 91, 34)
    vac = np.diff(np.concatenate([np.zeros((vac.shape[0], 1, vac.shape[2])), np.cumsum(vac, 1)], 1)[:, ::7], axis=1)
    vac = (vac - vac.mean(axis=0)) / (vac.std(axis=0) + 1e-8)
    vac = np.array(vac, dtype=np.float32)
    
    # hos normalization -> weekly (2334, 91, 4)
    hos = np.diff(np.concatenate([np.zeros((hos.shape[0], 1, hos.shape[2])), np.cumsum(hos, 1)], 1)[:, ::7], axis=1)
    hos = (hos - hos.mean(axis=0)) / (hos.std(axis=0) + 1e-8)
    hos = np.array(hos, dtype=np.float32)
    
    # claim normalization -> weekly (2334, 91, 20)
    claim = np.diff(np.concatenate([np.zeros((claim.shape[0], 1, claim.shape[2])), np.cumsum(claim, 1)], 1)[:, ::7], axis=1)
    claim = (claim - claim.mean(axis=0)) / (claim.std(axis=0) + 1e-8)
    claim = np.array(claim, dtype=np.float32)
    
    # distance normalization
    distance = (distance - distance.mean(axis=0)) / (distance.std(axis=0) + 1e-8)
    distance = np.array(distance, dtype=np.float32)
    
    # mobility normalization
    mob = (mob - mob.mean(axis=0)) / (mob.std(axis=0) + 1e-8)
    mob = np.array(mob, dtype=np.float32)
    
    # print the feature size
    print (" ------ feature size ------ ")
    print ("county", county.shape)
    print ("covid", covid.shape)
    print ("distance", distance.shape)
    print ("mob", mob.shape)
    print ("vac", vac.shape)
    print ("hos", hos.shape)
    print ("claim", claim.shape)
    print ("bron", bron.shape)
    print ("weeks", len(weeks))
    
    """
    ['2017-12-30', '2018-01-06', '2018-01-13', '2018-01-20', '2018-01-27', '2018-02-03', '2018-02-10', '2018-02-17', '2018-02-24', '2018-03-03', '2018-03-10', '2018-03-17', '2018-03-24', '2018-03-31', '2018-04-07', '2018-04-14', '2018-04-21', '2018-04-28', '2018-05-05', '2018-05-12', '2018-05-19', '2018-05-26', '2018-06-02', '2018-06-09', '2018-06-16', '2018-06-23', '2018-06-30', '2018-07-07', '2018-07-14', '2018-07-21', '2018-07-28', '2018-08-04', '2018-08-11', '2018-08-18', '2018-08-25', '2018-09-01', '2018-09-08', '2018-09-15', '2018-09-22', '2018-09-29', '2018-10-06', '2018-10-13', '2018-10-20', '2018-10-27', '2018-11-03', '2018-11-10', '2018-11-17', '2018-11-24', '2018-12-01', '2018-12-08', '2018-12-15', '2018-12-22', '2018-12-29', '2019-01-05', '2019-01-12', '2019-01-19', '2019-01-26', '2019-02-02', '2019-02-09', '2019-02-16', '2019-02-23', '2019-03-02', '2019-03-09', '2019-03-16', '2019-03-23', '2019-03-30', '2019-04-06', '2019-04-13', '2019-04-20', '2019-04-27', '2019-05-04', '2019-05-11', '2019-05-18', '2019-05-25', '2019-06-01', '2019-06-08', '2019-06-15', '2019-06-22', '2019-06-29', '2019-07-06', '2019-07-13', '2019-07-20', '2019-07-27', '2019-08-03', '2019-08-10', '2019-08-17', '2019-08-24', '2019-08-31', '2019-09-07', '2019-09-14', '2019-09-21', '2019-09-28', '2019-10-05', '2019-10-12', '2019-10-19', '2019-10-26', '2019-11-02', '2019-11-09', '2019-11-16', '2019-11-23', '2019-11-30', '2019-12-07', '2019-12-14', '2019-12-21', '2019-12-28', '2020-01-04', '2020-01-11', '2020-01-18', '2020-01-25', '2020-02-01', '2020-02-08', '2020-02-15', '2020-02-22', '2020-02-29', '2020-03-07', '2020-03-14', '2020-03-21', '2020-03-28', '2020-04-04', '2020-04-11', '2020-04-18', '2020-04-25', '2020-05-02', '2020-05-09', '2020-05-16', '2020-05-23', '2020-05-30', '2020-06-06', '2020-06-13', '2020-06-20', '2020-06-27', '2020-07-04', '2020-07-11', '2020-07-18', '2020-07-25', '2020-08-01', '2020-08-08', '2020-08-15', '2020-08-22', '2020-08-29', '2020-09-05', '2020-09-12', '2020-09-19', '2020-09-26', '2020-10-03', '2020-10-10', '2020-10-17', '2020-10-24', '2020-10-31', '2020-11-07', '2020-11-14', '2020-11-21', '2020-11-28', '2020-12-05', '2020-12-12', '2020-12-19', '2020-12-26', '2021-01-02', '2021-01-09', '2021-01-16', '2021-01-23', '2021-01-30', '2021-02-06', '2021-02-13', '2021-02-20', '2021-02-27', '2021-03-06', '2021-03-13', '2021-03-20', '2021-03-27', '2021-04-03', '2021-04-10', '2021-04-17', '2021-04-24', '2021-05-01', '2021-05-08', '2021-05-15', '2021-05-22', '2021-05-29', '2021-06-05', '2021-06-12', '2021-06-19', '2021-06-26', '2021-07-03', '2021-07-10', '2021-07-17', '2021-07-24', '2021-07-31', '2021-08-07', '2021-08-14', '2021-08-21', '2021-08-28', '2021-09-04', '2021-09-11', '2021-09-18', '2021-09-25', '2021-10-02', '2021-10-09', '2021-10-16', '2021-10-23', '2021-10-30', '2021-11-06', '2021-11-13', '2021-11-20', '2021-11-27', '2021-12-04', '2021-12-11', '2021-12-18', '2021-12-25', '2022-01-01', '2022-01-08', '2022-01-15', '2022-01-22', '2022-01-29', '2022-02-05', '2022-02-12', '2022-02-19', '2022-02-26', '2022-03-05', '2022-03-12', '2022-03-19', '2022-03-26', '2022-04-02', '2022-04-09', '2022-04-16', '2022-04-23', '2022-04-30', '2022-05-07', '2022-05-14', '2022-05-21', '2022-05-28', '2022-06-04', '2022-06-11', '2022-06-18', '2022-06-25', '2022-07-02', '2022-07-09', '2022-07-16', '2022-07-23', '2022-07-30', '2022-08-06', '2022-08-13', '2022-08-20', '2022-08-27', '2022-09-03', '2022-09-10', '2022-09-17', '2022-09-24', '2022-10-01', '2022-10-08', '2022-10-15', '2022-10-22', '2022-10-29', '2022-11-05', '2022-11-12', '2022-11-19', '2022-11-26', '2022-12-03', '2022-12-10', '2022-12-17', '2022-12-24', '2022-12-31', '2023-01-07', '2023-01-14']
    """
    
    val_start = weeks.index(args.val_start) # the val start index
    test_start = weeks.index(args.test_start) # the test start index

    predicted = []
    gt = []
    for i in tqdm(range(len(bron))):
        model = auto_arima(bron[i, :val_start, 11], error_action='ignore', suppress_warnings=True, seasonal=False)
        n_periods = bron.shape[1]-val_start
        cur_pred = model.predict(n_periods=n_periods)
        gt.append(bron[i, test_start:, 11])
        predicted.append(cur_pred[bron.shape[1]-test_start+1:])

    predicted = np.array(predicted)
    gt = np.array(gt)
    
    y_pred = predicted.flatten()
    y_true = gt.flatten()
    
    new_y_true = []
    new_y_pred = []
    for t, i in zip(y_true, y_pred):
        if t != 0:
            new_y_true.append(t)
            new_y_pred.append(i)
    y_true = np.array(new_y_true)
    y_pred = np.array(new_y_pred)

    mse_loss = np.mean((y_pred - y_true)**2)
    mae_loss = np.mean(np.abs(y_pred - y_true))
    mape_loss = mean_absolute_percentage_error(y_true, y_pred)
    ccc_loss = ccc(y_true, y_pred)
    rsquare_loss = r_squared(y_true, y_pred)
    pcc_loss = pcc(y_true, y_pred)
    
    print (f"===== ARIMA val_start {args.val_start}, test_start {args.test_start} ======")
    with open("../logs/results/arima2.log", "a") as f:
        print ({'mse': mse_loss, 'mae': mae_loss, 'mape': mape_loss, 'ccc': ccc_loss, 'rsquare': rsquare_loss, 'pcc': pcc_loss}, file=f)
        print ({'mse': mse_loss, 'mae': mae_loss, 'mape': mape_loss, 'ccc': ccc_loss, 'rsquare': rsquare_loss, 'pcc': pcc_loss}, file=f)
    print ("=====================")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--val_start", type=str, default='2022-11-26', help="the start of val window")
    parser.add_argument("--test_start", type=str, default='2022-12-24', help="the start of test window")
    parser.add_argument("--obs_window", type=int, default=120, help="num of weeks for observation")
    args = parser.parse_args()
    
    # baseline performance
    # {'mse': 77.90853, 'mae': 4.439636, 'mape': 1.4616153, 'ccc': 0.7552102113343029, 'rsquare': -0.5787722483290256, 'pcc': 0.7552101676566804}
    prepare_dataloader(args)
    
# ===== ARIMA ======
# {'mse': 70.3231811998278, 'mae': 3.7782290676388586, 'mape': 1.6208926186630213, 'ccc': -0.002875064736986032, 'rsquare': -6.399426071180202, 'pcc': -0.0028750643750832194}
# =====================