import pickle
import os
import argparse

import numpy as np
import torch
import  torch.nn as nn
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from utils import load_dataset, mean_absolute_percentage_error, ccc, r_squared, pcc, RSVLoader, collate_fn
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
    
    """ build training / val / test set """
    train_X, train_cty, train_Y, train_month = [], [], [], []
    for i in range(2334):
        if bron[i, :, 11].sum() == 0: continue
        for j in range(args.obs_window, val_start+4-args.test_length):
            # if sum(bron[i, j:j+4, 11]) == 0: continue
            train_X.append(bron[i, j-args.obs_window:j])
            train_cty.append(i)
            train_Y.append(bron[i, j:j+args.test_length, 11]) # 11 means the RSV cases
            train_month.append(int(weeks[j].split("-")[1]))
    
    val_X, val_cty, val_Y, val_month = [], [], [], []
    for i in range(2334):
        if bron[i, :, 11].sum() == 0: continue
        # if sum(bron[i, val_start:val_start+4, 11]) == 0: continue
        val_X.append(bron[i, val_start-args.obs_window:val_start])
        val_cty.append(i)
        val_Y.append(bron[i, val_start+4-args.test_length:val_start+4, 11]) # 11 means the RSV cases
        val_month.append(int(weeks[val_start].split("-")[1]))
    
    test_X, test_cty, test_Y, test_month = [], [], [], []
    for  i in range(2334):
        if bron[i, :, 11].sum() == 0: continue
        # if sum(bron[i, test_start:test_start+4, 11]) == 0: continue
        test_X.append(bron[i, test_start-args.obs_window:test_start])
        test_cty.append(i)
        test_Y.append(bron[i, test_start:test_start+args.test_length, 11]) # 11 means the RSV cases
        test_month.append(int(weeks[j].split("-")[1]))
            
    batch_size = 256
    train_loader = torch.utils.data.DataLoader(RSVLoader(county, covid, distance, mob, vac, hos, claim, county_tensor_emb, train_X, train_cty, train_month, train_Y), 
                    batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = torch.utils.data.DataLoader(RSVLoader(county, covid, distance, mob, vac, hos, claim, county_tensor_emb, val_X, val_cty, val_month, val_Y), 
                    batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(RSVLoader(county, covid, distance, mob, vac, hos, claim, county_tensor_emb, test_X, test_cty, test_month, test_Y), 
                    batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader
    
    
def clean_up_feature(batch):
        county, covid, distance, mob, vac, hos, claim, _, X_mtx, _, month, Y = batch
        """
        county torch.Size([256, 14])
        covid torch.Size([256, 91])
        distance torch.Size([256, 2334])
        mob torch.Size([256, 2334])
        vac torch.Size([256, 91, 34])
        hos torch.Size([256, 91, 4])
        claim torch.Size([256, 91, 20])
        county_tensor_emb torch.Size([256, 8]) not use
        X_mtx torch.Size([256, 120, 19])
        mask torch.Size([256, 120])
        month torch.Size([256])
        Y torch.Size([256, 4])
        """
        # county information (batch_size, 14)
        county_part = county
        # covid volume (batch_size, 1)
        covid_colume_part = covid.mean(-1).unsqueeze(-1)
        # covid 1st order difference (batch_size, 1)
        covid_1st_order = torch.diff(covid, dim=-1).mean(-1).unsqueeze(-1)
        # covid 2nd order difference (batch_size, 1)
        covid_2nd_order = torch.diff(covid, dim=-1, n=2).mean(-1).unsqueeze(-1)
        # covid 3rd order difference (batch_size, 1)
        covid_3rd_order = torch.diff(covid, dim=-1, n=3).mean(-1).unsqueeze(-1)
        # mobility mean (batch_size, 1)
        mobility_mean_part = mob.mean(-1).unsqueeze(-1)
        # mobility std (batch_size, 1)
        mobility_std_part = mob.std(-1).unsqueeze(-1)
        # distance mean (batch_size, 1)
        distance_mean_part = distance.mean(-1).unsqueeze(-1)
        # mobility std (batch_size, 1)
        distance_std_part = distance.std(-1).unsqueeze(-1)
        # vac volumne (batch_size, 1)
        vac_colume_part = vac.sum(-1).mean(-1).unsqueeze(-1)
        # vac 1st order difference (batch_size, 1)
        vac_1st_order = torch.diff(vac.sum(-1), dim=-1).mean(-1).unsqueeze(-1)
        # vac 2nd order difference (batch_size, 1)
        vac_2nd_order = torch.diff(vac.sum(-1), dim=-1, n=2).mean(-1).unsqueeze(-1)
        # vac 3rd order difference (batch_size, 1)
        vac_3rd_order = torch.diff(vac.sum(-1), dim=-1, n=3).mean(-1).unsqueeze(-1)
        # hos volume (batch_size, 1)
        hos_colume_part = hos.sum(-1).mean(-1).unsqueeze(-1)
        # hos 1st order difference (batch_size, 1)
        hos_1st_order = torch.diff(hos.sum(-1), dim=-1).mean(-1).unsqueeze(-1)
        # hos 2nd order difference (batch_size, 1)
        hos_2nd_order = torch.diff(hos.sum(-1), dim=-1, n=2).mean(-1).unsqueeze(-1)
        # hos 3rd order difference (batch_size, 1)
        hos_3rd_order = torch.diff(hos.sum(-1), dim=-1, n=3).mean(-1).unsqueeze(-1)
        # claims volume (batch_size, 1)
        claim_colume_part = claim.sum(-1).mean(-1).unsqueeze(-1)
        # claims 1st order difference (batch_size, 1)
        claim_1st_order = torch.diff(claim.sum(-1), dim=-1).mean(-1).unsqueeze(-1)
        # claims 2nd order difference (batch_size, 1)
        claim_2nd_order = torch.diff(claim.sum(-1), dim=-1, n=2).mean(-1).unsqueeze(-1)
        # claims 3rd order difference (batch_size, 1)
        claim_3rd_order = torch.diff(claim.sum(-1), dim=-1, n=3).mean(-1).unsqueeze(-1)
        # X volume (batch_size, 1)
        x_part = X_mtx.sum(-1).mean(-1).unsqueeze(-1)
        # X 1st order difference (batch_size, 1)
        x_1st_order = torch.diff(X_mtx.sum(-1), dim=-1).mean(-1).unsqueeze(-1)
        # X 2nd order difference (batch_size, 1)
        x_2nd_order = torch.diff(X_mtx.sum(-1), dim=-1, n=2).mean(-1).unsqueeze(-1)
        # X 3rd order difference (batch_size, 1)
        x_3rd_order = torch.diff(X_mtx.sum(-1), dim=-1, n=3).mean(-1).unsqueeze(-1)
        # month (batch_size, 1)
        month_part = month.unsqueeze(-1)
        
        # get X and Y
        X = torch.cat([county_part, covid_colume_part, covid_1st_order, covid_2nd_order, \
            covid_3rd_order, mobility_mean_part, mobility_std_part, distance_mean_part, distance_std_part, \
                vac_colume_part, vac_1st_order, vac_2nd_order, vac_3rd_order, hos_colume_part, hos_1st_order,\
                    hos_2nd_order, hos_3rd_order, claim_colume_part, claim_1st_order, claim_2nd_order, \
                        claim_3rd_order, x_part, x_1st_order, x_2nd_order, x_3rd_order, month_part], dim=-1)
        
        # X size: (batch_size, 39)
        # Y size: (batch_size, 4)
        return X, Y
        
        
def supervised(args):
    train_loader, val_loader, test_loader = prepare_dataloader(args)
    
    """
    form the training (train + val) and test set for XGBoost
    """
    X_train, y_train = [], []
    for batch in tqdm(train_loader):
        X, Y = clean_up_feature(batch)
        X_train.append(X)
        y_train.append(Y)
    
    X_val, y_val = [], []
    for batch in tqdm(val_loader):
        X, Y = clean_up_feature(batch)
        X_val.append(X)
        y_val.append(Y)
    
    X_test, y_test = [], []
    for batch in tqdm(test_loader):
        X, Y = clean_up_feature(batch)
        X_test.append(X)
        y_test.append(Y)
        
    X_train = torch.cat(X_train, dim=0).numpy()
    y_train = torch.cat(y_train, dim=0).numpy()
    X_val = torch.cat(X_val, dim=0).numpy()
    y_val = torch.cat(y_val, dim=0).numpy()
    X_test = torch.cat(X_test, dim=0).numpy()
    y_test = torch.cat(y_test, dim=0).numpy()
    
    print (f"===== XGBoost val_start {args.val_start}, test_start {args.test_start} ======")
    print (f"X_train shape: {X_train.shape}")
    print (f"Y_train shape: {y_train.shape}")
    print (f"X_val shape: {X_val.shape}")
    print (f"Y_val shape: {y_val.shape}")
    print (f"X_test shape: {X_test.shape}")
    print (f"Y_test shape: {y_test.shape}")
    
    # Define the XGBoost regression model
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                    max_depth = 5, alpha = 10, n_estimators = 10)

    # Define the parameter grid for hyperparameter tuning
    params = {
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [10, 50, 100],
            'max_depth': [2, 3, 5],
            'min_child_weight': [1, 2, 3]
            }

    # # Perform a grid search with cross-validation to find the best hyperparameters
    # grid = GridSearchCV(xg_reg, params, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=1)
    # grid.fit(X_train, y_train)

    # # Train the model with the best hyperparameters and early stopping
    # best_model = grid.best_estimator_
    xg_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # Predict the values for the test set
    y_pred = xg_reg.predict(X_test)
    
    y_pred = y_pred.flatten()
    y_true = y_test.flatten()
    
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
    
    # dump
    with open("../logs/results/xgboost.log", "a") as f:
        print ({'mse': mse_loss, 'mae': mae_loss, 'mape': mape_loss, 'ccc': ccc_loss, 'rsquare': rsquare_loss, 'pcc': pcc_loss}, file=f)
        print ({'mse': mse_loss, 'mae': mae_loss, 'mape': mape_loss, 'ccc': ccc_loss, 'rsquare': rsquare_loss, 'pcc': pcc_loss}, file=f)
    print ("=====================")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--val_start", type=str, default='2022-11-26', help="the start of val window")
    parser.add_argument("--test_start", type=str, default='2022-12-24', help="the start of test window")
    parser.add_argument("--test_length", type=int, default=4, help="the length of test window")
    parser.add_argument("--obs_window", type=int, default=120, help="num of weeks for observation")
    args = parser.parse_args()
    
    supervised(args)