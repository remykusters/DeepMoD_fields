import numpy as np
import torch
import time
import torch.nn as nn
import torch
from torch.autograd import grad


def Train_PINN(model, data, target, optimizer, max_iterations):
    
    print('| Iteration | Progress | Time remaining | Cost | MSE | Reg | L1')
    for iteration in torch.arange(0, max_iterations + 1):
        
        # Calculating prediction and library
        prediction, time_deriv, theta, coeff_vector = model(data)   
        rhs = time_deriv - theta*coeff_vector
       # print(rhs.shape)
       # print(theta.shape)
        # Calculating loss
        loss_reg = torch.mean((torch.sum(time_deriv - theta*coeff_vector,axis=1))**2)
        loss_mse = torch.mean((prediction - target)**2, dim=0)
        loss = loss_mse + loss_reg
        
        # Writing
        if iteration % 200 == 0:
            print("%.2f"% iteration, "%.2e"%loss.item(), "%.2e"%loss_mse.item())

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()