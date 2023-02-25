import os
import json
import numpy as np

if __name__ == "__main__":
    root = "../logs/results-exp1/"
    files = ["lstm", "transformer", "xgboost", "stan", "hoist", "taprsv"]
    
    # store the results of one model in to model = []
    # > model = [dict1, dict2, ...]
    # > models = [model1, model2, ...]
    
    # load all dicts
    models = []
    for file_name in files:
        file_path = os.path.join(root, file_name + ".log")
        models.append([json.loads(item[:-1].replace("'", '"')) for item in open(file_path, "r").readlines()[::2]])
    
    # compute window by window
    for w in range(4): # 4 windows
        print ()
        for model in models:
            # MSE
            mse = []
            for item in model[w*5: w*5+5]:
                if "mse" in item:
                    mse.append(item["mse"])
                else:
                    mse.append(item["test_mse_loss"])
            print ("{:.02f} ± {:.04f},".format(np.mean(mse), np.std(mse)), end="")
            
            # MAE
            mae = []
            for item in model[w*5: w*5+5]:
                if "mae" in item:
                    mae.append(item["mae"])
                else:
                    mae.append(item["test_mae_loss"])
            print ("{:.02f} ± {:.04f},".format(np.mean(mae), np.std(mae)), end="")
            
            # PCC
            pcc = []
            for item in model[w*5: w*5+5]:
                if "pcc" in item:
                    pcc.append(item["pcc"])
                else:
                    pcc.append(item["test_pcc_loss"])
            print ("{:.02f} ± {:.04f},".format(np.mean(pcc), np.std(pcc)), end="")
            
            # R2
            r2 = []
            for item in model[w*5: w*5+5]:
                if "rsquare" in item:
                    r2.append(item["rsquare"])
                else:
                    r2.append(item["test_rsquare_loss"])
            print ("{:.02f} ± {:.04f}".format(np.mean(r2), np.std(r2)))
            
    # in the end, use this https://tableconvert.com/csv-to-csv