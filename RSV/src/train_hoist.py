import pickle
import os
import argparse

import numpy as np
import torch
import  torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import HOIST
from utils import load_dataset, RSVLoader, collate_fn, mean_absolute_percentage_error, ccc, r_squared, pcc

class LitModel(pl.LightningModule):
    def __init__(self, args, disease_tensor_emb):
        super().__init__()
        self.args = args
        self.criterion = nn.MSELoss(reduction='none')
        hidden = 32
        self.model = HOIST(dynamic_dims=[19], static_dims=[hidden]*7, distance_dims=None, rnn_dim=32, hidden=hidden, signs=None, test_length=args.test_length)

    def training_step(self, batch, batch_idx):
        county, covid, distance, mob, vac, hos, claim, county_tensor_emb, X_mtx, mask, month, Y = batch
        prediction, _ = self.model([X_mtx], mask, [county, covid, vac, hos, claim, county_tensor_emb, month])
        Y = torch.log(Y+1)
        N, T, F = X_mtx.shape
        dist = _[0]
        weights = _[1]
        cur_d = X_mtx
        
        y_p = (weights * cur_d).sum(-1).reshape(N,T,1).mean(1)*prediction.detach()
        y_pi = y_p.reshape(N,1,args.test_length)
        y_pj = y_p.reshape(1,N,args.test_length)
        y_k = ((y_pi * y_pj) * dist.reshape(N,N,1)).sum(1).reshape(N,args.test_length,1)

        ising_loss = self.criterion((y_p.reshape(N,args.test_length,1)+y_k).squeeze(), Y.squeeze()).mean(1).mean()
        loss = self.criterion(prediction, Y).mean(-1).mean()# + ising_loss
        
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        county, covid, distance, mob, vac, hos, claim, county_tensor_emb, X_mtx, mask, month, Y = batch
        with torch.no_grad():
            prediction, _ = self.model([X_mtx], mask, [county, covid, vac, hos, claim, county_tensor_emb, month])
            step_result = torch.exp(prediction.cpu()).numpy()-1
            step_result[step_result<0] = 0
            step_gt = Y.cpu().numpy()
        return step_result, step_gt
    
    def validation_epoch_end(self, val_step_outputs):
        predicted = []
        gt =[]
        for out in val_step_outputs:
            predicted.append(out[0])
            gt.append(out[1])
        
        predicted = np.concatenate(predicted, axis=0)
        gt = np.concatenate(gt, axis=0)
        y_pred = np.concatenate(predicted, axis=0).flatten()
        y_true = np.concatenate(gt, axis=0).flatten()
        
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
        
        self.log("val_mse_loss", mse_loss, sync_dist=True)
        self.log("val_mae_loss", mae_loss, sync_dist=True)
        self.log("val_mape_loss", mape_loss, sync_dist=True)
        self.log("val_ccc_loss", ccc_loss, sync_dist=True)
        self.log("val_rsquare_loss", rsquare_loss, sync_dist=True)
        self.log("val_pcc_loss", pcc_loss, sync_dist=True)
        
    def test_step(self, batch, batch_idx):
        county, covid, distance, mob, vac, hos, claim, county_tensor_emb, X_mtx, mask, month, Y = batch
        with torch.no_grad():
            prediction, _ = self.model([X_mtx], mask, [county, covid, vac, hos, claim, county_tensor_emb, month])
            step_result = torch.exp(prediction.cpu()).numpy()-1
            step_result[step_result<0] = 0
            step_gt = Y.cpu().numpy()
        return step_result, step_gt

    def test_epoch_end(self, test_step_outputs):
        predicted = []
        gt =[]
        for idx, out in enumerate(test_step_outputs):
            predicted.append(out[0])
            gt.append(out[1])
        predicted = np.concatenate(predicted, axis=0)
        gt = np.concatenate(gt, axis=0)

        y_pred = np.concatenate(predicted, axis=0).flatten()
        y_true = np.concatenate(gt, axis=0).flatten()
        
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
        
        self.log("test_mse_loss", mse_loss, sync_dist=True)
        self.log("test_mae_loss", mae_loss, sync_dist=True)
        self.log("test_mape_loss", mape_loss, sync_dist=True)
        self.log("test_ccc_loss", ccc_loss, sync_dist=True)
        self.log("test_rsquare_loss", rsquare_loss, sync_dist=True)
        self.log("test_pcc_loss", pcc_loss, sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr, weight_decay=1e-5
        )
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        return [optimizer]#, [scheduler]
   

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
    bron = np.array(bron, dtype=np.float32)

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
                    batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=32, drop_last=True)
    val_loader = torch.utils.data.DataLoader(RSVLoader(county, covid, distance, mob, vac, hos, claim, county_tensor_emb, val_X, val_cty, val_month, val_Y), 
                    batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=32)
    test_loader = torch.utils.data.DataLoader(RSVLoader(county, covid, distance, mob, vac, hos, claim, county_tensor_emb, test_X, test_cty, test_month, test_Y), 
                    batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=32)
    
    print ("size of training", len(train_loader))
    print ("size of validation", len(val_loader))
    print ("size of testing", len(test_loader))

        
    """
    Measure the baseline performance:
        - we use the nearest month as the prediction
    """
    predicted = np.array([item[-4:, 11] for item in test_X])
    gt = np.array(test_Y)
    
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
    
    print ("===== standard ======")
    print ({'mse': mse_loss, 'mae': mae_loss, 'mape': mape_loss, 'ccc': ccc_loss, 'rsquare': rsquare_loss, 'pcc': pcc_loss})
    print ("=====================")
    return train_loader, val_loader, test_loader, disease_tensor_emb


def supervised(args):
    
    # get data loaders
    train_loader, val_loader, test_loader, disease_tensor_emb = prepare_dataloader(args)
    
    # folder to dump tensorboard logs
    root = "/home/chaoqiy2/github/multifaceted-graph-tensor/RSV/logs"
    
    # in case the log is over-written, this version name will change after each run
    version_id = len(os.listdir(os.path.join(root, "RSV-analysis"))) + 1
    version = f"HOIST-{args.val_start}-{args.test_start}-len-{args.test_length}-{version_id}"
        
    logger = TensorBoardLogger(
        save_dir=root,
        version=version,
        name="RSV-analysis",
    )
    
    # add early stop callback
    early_stop_callback = EarlyStopping(monitor="val_rsquare_loss", patience=3, verbose=False, mode="max")
    
    trainer = pl.Trainer(
        devices=[0, 1],
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        auto_select_gpus=True,
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback],
    )

    # define the model
    model = LitModel(args, disease_tensor_emb)
    
    # train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # test the model
    result = trainer.test(ckpt_path="best", dataloaders=test_loader)
    
    # dump
    with open("../logs/results/hoist.log", "a") as f:
        print (result[0], file=f)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--val_start", type=str, default='2022-11-26', help="the start of val window")
    parser.add_argument("--test_start", type=str, default='2022-12-24', help="the start of test window")
    parser.add_argument("--test_length", type=int, default=4, help="the length of test window")
    parser.add_argument("--obs_window", type=int, default=120, help="num of weeks for observation")
    args = parser.parse_args()
    
    # baseline performance
    # {'mse': 77.90853, 'mae': 4.439636, 'mape': 1.4616153, 'ccc': 0.7552102113343029, 'rsquare': -0.5787722483290256, 'pcc': 0.7552101676566804}
    # model performance:

    # test_ccc_loss         0.2744828442693468      
    # test_mae_loss          2.554976224899292
    # test_mape_loss         0.9223506450653076
    # test_mse_loss          47.10430908203125      
    # test_pcc_loss         0.2744828555940975
    # test_rsquare_loss       0.07007720474366808
    supervised(args)
    
  
# RNN 32 Hidden 16
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#        Test metric             DataLoader 0
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#       test_ccc_loss         0.7111108575594423
#       test_mae_loss         2.1736931800842285
#      test_mape_loss         1.0159484148025513
#       test_mse_loss          30.07427406311035
#       test_pcc_loss         0.7111108459354735
#     test_rsquare_loss       0.4452604592100674
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# RNN 64 Hidden 32
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#        Test metric             DataLoader 0
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#       test_ccc_loss         0.7608735568154699
#       test_mae_loss          2.179419755935669
#      test_mape_loss         1.0196350812911987
#       test_mse_loss         29.613494873046875
#       test_pcc_loss         0.7608736301969623
#     test_rsquare_loss       0.45375980682262307
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
