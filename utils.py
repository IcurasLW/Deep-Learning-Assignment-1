from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
from torch.nn import functional as F
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import csv
import os



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float32)
torch.cuda.manual_seed_all(0)





class Ourdataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx],
            'target': self.targets[idx]
        }
        return sample



def evaluate(y_pred, y_true):
    y_pred_prob = y_pred.copy()
    y_pred = np.where(y_pred>=0.5, 1, 0)
    acc = accuracy_score(y_pred=y_pred, y_true=y_true)
    f1 = f1_score(y_pred=y_pred, y_true=y_true, average='binary', zero_division=1)
    precision = precision_score(y_pred=y_pred, y_true=y_true, average='binary', zero_division=1)
    recall = recall_score(y_pred=y_pred, y_true=y_true, average='binary', zero_division=1)
    auc = roc_auc_score(y_true, y_pred_prob)
    output = {
            'Acc':acc,
            'F1': f1,
            'Precision': precision,
            'Recall': recall,
            'AUC': auc
            }

    return output



def data_prepare(data_path):
    '''
    This function uses the data has been preprocessed as indicated in guideline
    Input: DataPath
    Output: X, Y in numpy format
    '''
    
    df = pd.read_csv(data_path, sep=' ', header=None).drop(columns=[9]).dropna()
    Y = df.loc[:, 0].to_numpy()
    Y = np.where(Y == -1, 0, Y)
    fn_1 = lambda x: x.str[2:].astype('float32')
    df = df.iloc[:, 1:].apply(fn_1, axis=1)
    X = df.to_numpy()
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2,stratify=Y)
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.fit_transform(test_X)
    
    
    return train_X, test_X, train_Y, test_Y



def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)



def save_results(save_path, fold_num, metrics:dict, mode:str, model_name):
    '''
    Each row is one epoch
    '''
    output_filename =  save_path + f'{model_name}_fold_{fold_num}_{mode}_results.csv'

    # save train results
    try:
        # if the results exist
        with open (output_filename, 'r') as log:
            pass
        
        print('file exists')
        with open(output_filename, 'a+', newline='') as log:
            writer = csv.DictWriter(log, fieldnames=metrics.keys())
            writer.writerow(metrics)

    except:
        print('file not exists')
        create_folder(save_path)
        with open(output_filename, 'w', newline='') as log:
            writer = csv.DictWriter(log, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)
            