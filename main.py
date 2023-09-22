import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from utils import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from classifier import *
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float32)
torch.cuda.manual_seed_all(0)



def train(model, optimizer, dataloader, loss_fn, fold_num, model_name, epochs=10):
    model.train()
    save_path = 'output/'
    
    for e in range(epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        y_pred = []
        y_true = []
        losses = 0
        for batch_idx, batch in progress_bar:
            optimizer.zero_grad()
            output = model(batch['data'].to(DEVICE))
            loss = loss_fn(output, batch['target'].to(DEVICE).float())
            losses += loss.item()
            y_pred.append(output.detach().cpu().numpy())
            y_true.append(batch['target'].detach().cpu().numpy())
            loss.backward()
            optimizer.step()
        
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true)
        metrics = evaluate(y_pred, y_true)
        losses /= batch_idx + 1
        metrics['Loss'] = losses
        
        
        save_results(save_path, fold_num, metrics, mode='train', model_name=model_name)





def validate(model, dataloader, loss_fn, fold_num, model_name):
    model.eval()
    y_pred = []
    y_true = []
    losses = 0
    save_path = 'output/'
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            output = model(batch['data'].to(DEVICE))
            loss = loss_fn(output, batch['target'].to(DEVICE).float())
            losses += loss.item()
            y_pred.append(output.detach().cpu().numpy())
            y_true.append(batch['target'].detach().cpu().numpy())
            
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true)
    metrics = evaluate(y_pred, y_true)
    losses /= batch_idx + 1
    metrics['Loss'] = losses
    
    save_results(save_path, fold_num, metrics, model_name=model_name, mode='val')





def test(model,  dataloader, loss_fn, model_name):
    model.eval()
    y_pred = []
    y_true = []
    losses = 0
    save_path = 'output/'
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            output = model(batch['data'].to(DEVICE))
            loss = loss_fn(output, batch['target'].to(DEVICE).float())
            losses += loss.item()
            y_pred.append(output.detach().cpu().numpy())
            y_true.append(batch['target'].detach().cpu().numpy())

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true)
    metrics = evaluate(y_pred, y_true)
    losses /= batch_idx + 1
    metrics['Loss'] = losses
    
    
    save_results(save_path, 0, metrics, mode='test', model_name=model_name)



def cross_validate(data_path, k=5, epochs=20, batch_size=32):
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    
    # Preprocess data
    train_X, test_X, train_Y, test_Y = data_prepare(data_path)
    
    
    for f, (train_index, val_index) in enumerate(kf.split(train_X)):
        # Model Initialization
        model_MLP_V1 = MLP_V1(train_X.shape[1]).to(DEVICE)
        model_MLP_V2 = MLP_V2(train_X.shape[1]).to(DEVICE)
        model_SLP = SinglePerceptron(train_X.shape[1]).to(DEVICE)
        
        # Optimizer Initialization
        optimizer_MLP_V_2 = torch.optim.SGD(params=model_MLP_V2.parameters(), lr=1e-3, weight_decay=1e-4)
        optimizer_MLP_V_1 = torch.optim.SGD(params=model_MLP_V1.parameters(), lr=1e-3, weight_decay=1e-4)
        optimizer_SLP = torch.optim.Adam(params=model_SLP.parameters(), lr=0.1, weight_decay=1e-3)
        
        # Loss function Initialization
        loss_fn = nn.BCELoss()


        # Data split
        train_X_k, val_X = train_X[train_index], train_X[val_index]
        train_Y_k, val_Y = train_Y[train_index], train_Y[val_index]
        
        train_dataset = Ourdataset(train_X_k, train_Y_k)
        val_dataset = Ourdataset(val_X, val_Y)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        
        
        
        ########### Train & Validate Phrase #############
        train(model_MLP_V1, optimizer_MLP_V_1, train_dataloader, loss_fn, f, epochs=epochs, model_name='MLP_V1')
        validate(model_MLP_V1, val_dataloader, loss_fn, f, model_name='MLP_V1')
        
        train(model_MLP_V2, optimizer_MLP_V_2, train_dataloader, loss_fn, f, epochs=epochs, model_name='MLP_V2')
        validate(model_MLP_V2, val_dataloader, loss_fn, f, model_name='MLP_V2')
        
        train(model_SLP, optimizer_SLP, train_dataloader, loss_fn, f, epochs=epochs, model_name='SLP')
        validate(model_SLP, val_dataloader, loss_fn, f, model_name='SLP')
        RF_model, SVM_model = stats_CV(train_X_k, val_X, train_Y_k, val_Y, k=f)
    
    
    test_dataset = Ourdataset(test_X, test_Y)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    stats_test(RF_model, test_X, test_Y, model_name='RF')
    stats_test(SVM_model, test_X, test_Y, model_name='SVM')
    test(model_MLP_V1, test_dataloader, loss_fn, model_name='MLP_V1')
    test(model_MLP_V2, test_dataloader, loss_fn, model_name='MLP_V2')
    test(model_SLP, test_dataloader, loss_fn, model_name='SLP')
    


    # save the test data and model for drawing graphs in jupyter notebook
    np.save('./models/test_X.npy', test_X)
    np.save('./models/test_Y.npy', test_Y)
    torch.save(model_MLP_V1.state_dict(), './models/model_MLP_V1.pth')
    torch.save(model_MLP_V2.state_dict(), './models/model_MLP_V2.pth')
    torch.save(model_SLP.state_dict(), './models/model_SLP.pth')
    joblib.dump(RF_model, './models/RF_model.pkl')
    joblib.dump(SVM_model, './models/SVM_model.pkl')
    



###################################### Statistical Model ######################################
def stats_test(model, test_X, test_Y, model_name):
    y_pred = model.predict(test_X)
    metrics = evaluate(y_pred, test_Y)
    save_path = 'output/'
    save_results(save_path, model_name, metrics, mode='test', model_name=model_name)



def stats_CV(train_X, test_X, train_Y, test_Y, k):
    RF_model = RandomForestClassifier(n_estimators=10, random_state=0)
    RF_model.fit(train_X, train_Y)
    y_pred_rf = RF_model.predict(test_X)
    
    SVM_model = SVC(kernel='rbf', C=10, random_state=0)
    SVM_model.fit(train_X, train_Y)
    y_pred_svm = SVM_model.predict(test_X)

    rf_metrics = evaluate(y_pred_rf, test_Y)
    svm_metrics = evaluate(y_pred_svm, test_Y)
    save_path = 'output/'
    save_results(save_path, k, rf_metrics, mode='val', model_name='RF')
    save_results(save_path, k, svm_metrics, mode='val', model_name='SVM')
    return RF_model, SVM_model


if __name__ == "__main__":
    data_path = 'diabetes_scaled.txt'
    cross_validate(data_path=data_path)