import pickle
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from scipy import stats

class RSVLoader(torch.utils.data.Dataset):
    """
    county (2334, 14)
    covid (2334, 639)
    distance (2334, 2334)
    mob (2334, 2334)
    vac (2334, 639, 34)
    hos (2334, 639, 4)
    claim (2334, 639, 20)
    county_tensor_emb (2334, 8): county tensor representation
    X: a list of samples of size (observe window, 19)
    cty: the county index of each sample
    month: the month index of each sample
    Y: a list of targets of size (4,)
    """
    
    def __init__(self, county, covid, distance, mob, vac, hos, claim, county_tensor_emb, X, cty, month, Y):
        self.county = county
        self.covid = covid
        self.distance = distance
        self.mob = mob
        self.vac = vac
        self.hos = hos
        self.claim = claim
        self.county_tensor_emb = county_tensor_emb
        self.X = X
        self.cty = cty
        self.Y = Y
        self.month = month

    def __len__(self):
        return len(self.cty)

    def __getitem__(self, index):
        # get the county idx
        idx = self.cty[index]
        # get the size information for this samples
        county_tensor_emb = self.county_tensor_emb[idx]
        county = self.county[idx]
        covid = self.covid[idx]
        distance = self.distance[idx]
        mob = self.mob[idx]
        vac = self.vac[idx]
        hos = self.hos[idx]
        claim = self.claim[idx]
        # get month idx
        month = self.month[index]
        # get the observation window and target
        X = self.X[index]
        Y = self.Y[index]

        return county, covid, distance, mob, vac, hos, claim, county_tensor_emb, X, month, Y

class RSVLoader_STAN(torch.utils.data.Dataset):
    """
    county (2334, 14)
    covid (2334, 639)
    distance (2334, 2334)
    mob (2334, 2334)
    vac (2334, 639, 34)
    hos (2334, 639, 4)
    claim (2334, 639, 20)
    county_tensor_emb (2334, 8): county tensor representation
    X: a list of samples of size (observe window, 19)
    cty: the county index of each sample
    month: the month index of each sample
    Y: a list of targets of size (4,)
    """
    
    def __init__(self, county, covid, distance, mob, vac, hos, claim, county_tensor_emb, X, cty, month, Y):
        self.county = county
        self.covid = covid
        self.distance = distance
        self.mob = mob
        self.vac = vac
        self.hos = hos
        self.claim = claim
        self.county_tensor_emb = county_tensor_emb
        self.X = X
        self.cty = cty
        self.Y = Y
        self.month = month

    def __len__(self):
        return len(self.cty)

    def __getitem__(self, index):
        # get the county idx
        idx = self.cty[index]
        # get the size information for this samples
        county_tensor_emb = self.county_tensor_emb[idx]
        county = self.county[idx]
        covid = self.covid[idx]
        distance = self.distance[idx]
        mob = self.mob[idx]
        vac = self.vac[idx]
        hos = self.hos[idx]
        claim = self.claim[idx]
        # get month idx
        month = self.month[index]
        # get the observation window and target
        X = self.X[index]
        Y = self.Y[index]

        return county, covid, distance, mob, vac, hos, claim, county_tensor_emb, X, month, Y, idx

   
def collate_fn(batch):
    """ data is a list of sample from the dataset """
    county, covid, distance, mob, vac, hos, claim, county_tensor_emb, X, month, Y = zip(*batch)
    county = torch.FloatTensor(np.array(county)) # (batch_size, 14)
    covid = torch.FloatTensor(np.array(covid)) # (batch_size, 639)
    distance = torch.FloatTensor(np.array(distance)) # (batch_size, 2334)
    mob = torch.FloatTensor(np.array(mob)) # (batch_size, 2334)
    vac = torch.FloatTensor(np.array(vac)) # (batch_size, 639, 34)
    hos = torch.FloatTensor(np.array(hos)) # (batch_size, 639, 4)
    claim = torch.FloatTensor(np.array(claim)) # (batch_size, 639, 20)
    county_tensor_emb = torch.FloatTensor(np.array(county_tensor_emb)) # (batch_size, 8)
    max_obs_len = max([x.shape[0] for x in X])
    X_mtx = torch.zeros(len(X), max_obs_len, 19) # (batch_size, max_obs_len, 19)
    mask = torch.zeros(len(X), max_obs_len) # (batch_size, max_obs_len)
    for idx, x in enumerate(X):
        X_mtx[idx, :x.shape[0], :] = torch.FloatTensor(x)
        mask[idx, :x.shape[0]] = 1
    mask = torch.BoolTensor(mask.bool())
    month = torch.LongTensor(month)  # (batch_size,)
    Y = torch.FloatTensor(np.array(Y)) # (batch_size, 4)
    
    return county, covid, distance, mob, vac, hos, claim, county_tensor_emb, X_mtx, mask, month, Y
   
   

def collate_fn_stan(batch):
    """ data is a list of sample from the dataset """
    county, covid, distance, mob, vac, hos, claim, county_tensor_emb, X, month, Y, idx = zip(*batch)
    idx = np.array(idx)
    county = torch.FloatTensor(np.array(county)) # (batch_size, 14)
    covid = torch.FloatTensor(np.array(covid)) # (batch_size, 639)
    new_dis = np.array(distance)[:, idx] - np.array(mob)[:, idx]
    threshold = np.percentile(new_dis, 80)
    adj = torch.tensor((new_dis > threshold), dtype=torch.int)
    distance = torch.FloatTensor(np.array(distance))[:, idx] # (batch_size, 2334)
    mob = torch.FloatTensor(np.array(mob))[:, idx] # (batch_size, 2334)
    
    vac = torch.FloatTensor(np.array(vac)) # (batch_size, 639, 34)
    hos = torch.FloatTensor(np.array(hos)) # (batch_size, 639, 4)
    claim = torch.FloatTensor(np.array(claim)) # (batch_size, 639, 20)
    county_tensor_emb = torch.FloatTensor(np.array(county_tensor_emb)) # (batch_size, 8)
    max_obs_len = max([x.shape[0] for x in X])
    X_mtx = torch.zeros(len(X), max_obs_len, 19) # (batch_size, max_obs_len, 19)
    mask = torch.zeros(len(X), max_obs_len) # (batch_size, max_obs_len)
    for idx, x in enumerate(X):
        X_mtx[idx, :x.shape[0], :] = torch.FloatTensor(x)
        mask[idx, :x.shape[0]] = 1
    mask = torch.BoolTensor(mask.bool())
    month = torch.LongTensor(month)  # (batch_size,)
    Y = torch.FloatTensor(np.array(Y)) # (batch_size, 4)
    
    return county, covid, distance, mob, vac, hos, claim, county_tensor_emb, X_mtx, mask, month, Y, adj
   
   
def collate_fn_cola(batch):
    """ data is a list of sample from the dataset """
    county, covid, distance, mob, vac, hos, claim, county_tensor_emb, X, month, Y = zip(*batch)
    
    X_mtx = torch.zeros(len(X), X[0].shape[1], X[0].shape[0]) # (batch_size, max_obs_len, 19)
    mask = torch.zeros(len(X), X[0].shape[1]) # (batch_size, max_obs_len)
    for idx, x in enumerate(X):
        X_mtx[idx, :x.shape[1], :] = torch.FloatTensor(x).permute(1, 0)
        mask[idx, :x.shape[1]] = 1
    mask = torch.BoolTensor(mask.bool())
    Y = torch.FloatTensor(np.array(Y)) # (batch_size, 4)
    
    return X_mtx, mask, Y
   
def load_dataset():
    root = "../" # relative in the RSV folder
    county = pickle.load(open(os.path.join(root, "data/county.pkl"), "rb"))
    covid = pickle.load(open(os.path.join(root, "data/covid.pkl"), "rb"))
    distance = pickle.load(open(os.path.join(root, "data/distance.pkl"), "rb"))
    mob = pickle.load(open(os.path.join(root, "data/mob.pkl"), "rb"))
    vac = pickle.load(open(os.path.join(root, "data/vac.pkl"), "rb"))
    hos = pickle.load(open(os.path.join(root, "data/hos.pkl"), "rb"))
    claim = pickle.load(open(os.path.join(root, "data/claim.pkl"), "rb"))
    bron = pickle.load(open(os.path.join(root, "data/bron.pkl"), "rb")) # get it from notebook/IQVIA disease cluster.ipynb
    weeks = pickle.load(open(os.path.join(root, "data/weeks.pkl"), "rb")) # get it from notebook/IQVIA disease cluster.ipynb
    fips = pickle.load(open(os.path.join(root, "data/fips.pkl"), "rb"))
    
    # (disease, embedding dim)
    disease_tensor_emb = pickle.load(open(os.path.join(root, "data/disease_tensor_emb.pkl"), "rb")) # get it from notebook/IQVIA disease cluster.ipynb
    # (county, embedding dim)
    county_tensor_emb = pickle.load(open(os.path.join(root, "data/county_tensor_emb.pkl"), "rb")) # get it from notebook/IQVIA disease cluster.ipynb
    return county, covid, distance, mob, vac, hos, claim, bron, weeks, fips, disease_tensor_emb, county_tensor_emb


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    return np.mean((np.abs(y_true - y_pred) + 1) / (y_true + 1))

def pcc(y_true,y_pred):
    ''' Pearson Correlation Coefficient'''
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    sxy = np.sum((y_pred - y_pred.mean())*(y_true - y_true.mean()))/y_true.shape[0]
    rho = sxy / (np.std(y_pred)*np.std(y_true))
    return rho

def r_squared(y_true, y_pred, std=False):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    return r2_score(y_true, y_pred)

def ccc(y_true, y_pred, std=False):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    return stats.pearsonr(y_true, y_pred)[0]