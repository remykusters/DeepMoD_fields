import numpy as np
import torch
import time
import torch.nn as nn
import torch
from torch.autograd import grad


class DeepMoD_fields(nn.Module):
    def __init__(self, n_in, hidden_dims, n_out, library_func, library_args, n_samples, n_outc, n_inc=1):
        super().__init__()
        self.network = self.build_network(n_in, hidden_dims, n_out)
        self.network_xi = self.build_network_xi(n_inc,hidden_dims,n_outc)
        self.library_func = library_func
        self.library_args = library_args
        
    def forward(self, input):
        # First network --> Calculate prediction + library 
        prediction = self.network(input)
        time_deriv, theta = self.library_func((prediction,input),**self.library_args)

        # Second network --> Calculates the spatial dependent coefficient vector
        x = input[:,1].reshape(-1,1)
        xi = self.network_xi(x)
        
        return prediction, time_deriv, theta, xi
      
    def build_network(self, n_in, hidden_dims, n_out):
        network = []
        hs = [n_in] + hidden_dims + [n_out]
        for h0, h1 in zip(hs, hs[1:]):  # Hidden layers
            network.append(nn.Linear(h0, h1))
            network.append(nn.Tanh())
        network.pop()  # get rid of last activation function
        network = nn.Sequential(*network) 
        return network
    
    def build_network_xi(self, n_inc, hidden_dims, n_outc):
        networkc = []
        hs = [n_inc] + hidden_dims + [n_outc]
        for h0, h1 in zip(hs, hs[1:]):  # Hidden layers
            networkc.append(nn.Linear(h0, h1))
            networkc.append(nn.Tanh())
        networkc.pop()  # get rid of last activation function
        networkc = nn.Sequential(*networkc) 
        return networkc    
    
    def network_parameters(self):
        return self.network.parameters()
    
    def network_parameters_xi(self):
        return self.network_xi.parameters()
