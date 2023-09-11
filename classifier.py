import torch
import torch.nn as nn


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float32)
torch.cuda.manual_seed_all(0)



class MLP_V1(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        self.extract = nn.Sequential(nn.Linear(in_features=input_size, out_features=32),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(32),
                                    
                                    nn.Linear(in_features=32, out_features=32),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(32),
                                    
                                    nn.Linear(in_features=32, out_features=32),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(32))
        
        self.fc = nn.Sequential(nn.Linear(in_features=32, out_features=1),
                                        nn.Sigmoid())

    def forward(self, X):
        output = self.extract(X)
        output = self.fc(output)
        return output.view(-1)



class MLP_V2(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        self.extract = nn.Sequential(nn.Linear(in_features=input_size, out_features=128),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(128),
                                    
                                    nn.Linear(in_features=128, out_features=128),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(128),
                                    
                                    nn.Linear(in_features=128, out_features=128),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(128))
        
        self.fc = nn.Sequential(nn.Linear(in_features=128, out_features=1),
                                        nn.Sigmoid())

    def forward(self, X):
        output = self.extract(X)
        output = self.fc(output)
        return output.view(-1)



class SinglePerceptron(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        self.perceptron = nn.Sequential(nn.Linear(in_features=input_size, out_features=1),
                                        nn.Sigmoid())
        
    
    def forward(self, X):
        output = self.perceptron(X)
        return output.view(-1)