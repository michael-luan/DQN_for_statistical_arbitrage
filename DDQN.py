# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:39:56 2022

@author: sebja
"""

from Environment import Environment 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

import torch
import torch.optim as optim
import torch.nn as nn

import seaborn as sb

from tqdm import tqdm

import copy



import pdb
def color(number):
    if number > 0:
        return "blue", 10
    elif number <0:
        return "red", -10
    else:
        return "black", 0
    
colorv = np.vectorize(color)

action_size = 21
class ANN(nn.Module):
    
    def __init__(self, n_in, n_out, nNodes, nLayers, activation='silu' ):
        super(ANN, self).__init__()
        
        self.prop_in_to_h = nn.Linear( n_in, nNodes)
        
        self.prop_h_to_h = nn.ModuleList([nn.Linear(nNodes, nNodes) for i in range(nLayers-1)])
            
        self.prop_h_to_out = nn.Linear(nNodes, n_out)
        
        if activation == 'silu':
            self.g = nn.SiLU()
        elif activation =='relu':
            self.g = nn.ReLU()
            
        # # log(1 + e^x) -- this is not needed (yet)
        # self.softplus = nn.Softplus()

    def forward(self, x):
        
        # input into  hidden layer
        h = self.g(self.prop_in_to_h(x))
        
        for prop in self.prop_h_to_h:
            h = self.g(prop(h))
        
        # hidden layer to output layer - no activation
        y = self.prop_h_to_out(h)
        
        return y

class DDQN():
    
    def __init__(self, env : Environment, gamma = 1, n_nodes = 20, n_layers = 5, lr=1e-3):
        
        self.env = env
        self.gamma = gamma
        
        # define a network for our Q-function approximation
        #
        # features = asset price, inventory
        # out = Q((S,I), a)  a = short, nothing, long
        #
        self.Q_main = ANN(n_in=2, n_out=action_size, nNodes=n_nodes, nLayers=n_layers)
        
        self.Q_target = copy.deepcopy(self.Q_main)
        
        # define an optimzer
        self.optimizer = optim.AdamW(self.Q_main.parameters(), lr)
        
        self.loss = []
        
        self.S = []
        self.I = []
        self.r = []
        self.epsilon = []
        
    

    
    def random_choice(self, a, size):
        
        b = torch.tensor(np.random.choice(a, size)).int()
        
        return b
    
    def action_idx_to_action(self, a_idx):
    
        return a_idx - 10
        
    def Train(self, n_iter = 10_000, mini_batch_size=256, n_plot=500):
        
        # ranomly initialize the initial conditions
        
        S = self.env.randomise_S0(mini_batch_size)
        S = torch.from_numpy(S).float()
        
        I = self.random_choice(list(range(-self.env.Q, self.env.Q+1)), (mini_batch_size,)).float()
        T = torch.ones(mini_batch_size).float() * self.env.T
        
        C = 100
        D = 100
        for i in tqdm(range(n_iter)):
        
            epsilon = C/(D+i)
            
            self.epsilon.append(epsilon)
        
            # this is for proper gradient computations
            self.optimizer.zero_grad()
            
            
            # concatenate states so net can compute properly
            X = torch.cat( (S.reshape(-1,1), I.reshape(-1,1)), axis=1)
            Q = self.Q_main(X)
            
            # find argmax for greedy actions
            _, a_idx = torch.max(Q, dim=1)
            
            # # add in epsilon-greedy actions
            H = (torch.rand(mini_batch_size) < epsilon)
            a_idx[H] = self.random_choice(np.arange(0, 21, step=1), torch.sum(H).numpy()).long()
            
            # import pdb
            # pdb.set_trace()
            
            Q_eval = torch.zeros(mini_batch_size).float()
            for k in range(21):
                mask = (a_idx == k)
                Q_eval[mask] = Q[mask, k]
            
            # import pdb
            # pdb.set_trace()
            
            a = self.action_idx_to_action(a_idx)
            
            
            # step in the environment
            Sp, Ip, T, r = self.env.step(S, I, T, a.detach(), nsims = mini_batch_size)
            
            # compute the Q(S', a*)
            Xp = torch.cat( (Sp.reshape(-1,1), Ip.reshape(-1,1)), axis=1)
            Qp_target = self.Q_target(Xp)
            Qp_main = self.Q_main(Xp)
            
            _, ap_idx = torch.max(Qp_main, dim=1)
            
            Qp_eval = torch.zeros(mini_batch_size).float()
            for k in range(21):
                mask = (ap_idx == k)
                Qp_eval[mask] = Qp_target[mask, k]
            
            # compute the loss
            target = r + self.gamma * Qp_eval
            loss = torch.mean(( target.detach() - Q_eval )**2)
         
            # compute the gradients
            loss.backward()
            
            # perform SGD / Adam  AdamW step using those gradients
            self.optimizer.step()
            
            self.loss.append(loss.item())
            
            # update state
            S = Sp.clone()
            I = Ip.clone()
            
            self.S.append(S.numpy())
            self.I.append(I.numpy())
            self.r.append(r.numpy())
            
            if np.mod(i,100) == 0:
                self.Q_target = copy.deepcopy(self.Q_main)
            
            if np.mod(i, n_plot) == 0:
                
                plt.plot(self.loss, linewidth = 0.5)
                plt.yscale('log')
                plt.title("Loss History Over Iterations", fontsize = 16)
                plt.xlabel("Iterations")
                plt.ylabel("Loss")
                plt.show()
                self.RunStrategy(100)
                
        self.Q_target = copy.deepcopy(self.Q_main)
        return self.RunStrategy(100)
        
        
    def RunStrategy(self, nsteps, nsims=1_000, mini_batch_size=256,):
        
        
        S = torch.zeros((nsims,nsteps)).float()
        I = torch.zeros((nsims,nsteps)).float()
        a = torch.zeros((nsims,nsteps)).float()
        r = torch.zeros((nsims,nsteps)).float()
        T = torch.ones(mini_batch_size).float() * self.env.T
        
        S[:,0] = self.env.theta
        I[:,0] = 0
        
        for i in range(nsteps-1):
            
            X = torch.cat((S[:,i].reshape(-1,1),I[:,i].reshape(-1,1)), axis=1)
            Q = self.Q_main(X).detach()
            
            a_idx = torch.argmax(Q, axis=1)
            
            a[:,i] = self.action_idx_to_action(a_idx)
            
            # import pdb
            # pdb.set_trace()
            
            S[:,i+1], I[:,i+1], T , r[:,i], = self.env.step(S[:,i], I[:,i], T, a[:,i], nsims = nsims)
            
            
        S = S.detach().numpy()
        a = a.detach().numpy()
        r = r.detach().numpy()
        I = I.detach().numpy()
        
        t = (self.env.T/nsteps)*np.arange(0,S.shape[1])
        
        plt.subplot(2,2,1)
        plt.plot(t, S[0,:])
        plt.plot(t, self.env.theta + (self.env.S0-self.env.theta)*np.exp(-self.env.kappa*t))
        plt.axhline(self.env.theta, linestyle='--', color='k')
        plt.title("Price Over Time", fontsize = 20)
        plt.ylabel(r"$S_t$" , fontsize = 16)
        
        plt.subplot(2,2,2)
        plt.plot(t, np.cumsum(r[0,:]))
        plt.title("Rewards Over Time", fontsize = 20)
        plt.ylabel(r"$\sum_{u=1}^t r_u$", fontsize = 16)
        
        plt.subplot(2,2,3)
        plt.plot(t, I[0,:])
        plt.title("Inventory Over Time", fontsize = 20)
        plt.ylabel(r"$I_t$", fontsize = 16)
        plt.xlabel("Days", fontsize = 16)
        
        plt.subplot(2,2,4)
        plt.plot(t, a[0,:])
        plt.title("Optimal Action Over Time", fontsize = 20)
        plt.ylabel(r"$a_t$" , fontsize = 16)
        plt.xlabel("Days",  fontsize = 16)
        
        plt.tight_layout()
        
        plt.show()
        
        plt.hist(np.sum(r,axis=1), bins=20, density=True)
        plt.title(r"Density of $\sum_t r_t$", fontsize = 16)
        plt.ylabel("Density")
        plt.xlabel("Expected Sum of Rewards")
        plt.show()
    

        
        # for i in range(0, 100, 10):
            
        #     plt.scatter(x = I[:, i], y = S[:, i], c = color(int(np.mean(a[:, i])))[0], alpha = 0.5, label = color(int(np.mean(a[:, i])))[1])
        #     # plt.title("Optimal Strategy at time " + str((i + 1)//10))
        #     plt.xlabel("Inventory")
        #     plt.ylabel("Price")
        #     print()
        # plt.legend()
        # plt.show()
        # print(I[0, i], a[0, i])
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        
        for i in range(100):
            data = np.array((I[:10, i], S[:10, i], i, a[:10, i]))
            x, y, z, act = data
            ax.scatter(x, y, z, c = colorv(act)[0])
        # fig.colorbar(p, ax=ax)
        plt.xlabel("Inventory", fontsize = 16)
        ax.set_ylabel("Price", fontsize = 16, y = 2)
        ax.set_zlabel("Day", fontsize = 16)
        
        
        red_patch = mpatches.Patch(color='red', label='-10')
        blue_patch = mpatches.Patch(color='blue', label='10')
        black_patch = mpatches.Patch(color='black', label='0/Liquidate')

        plt.legend(handles=[red_patch, blue_patch, black_patch])
        plt.title("Optimal Strategy Over Time, Inventory, and Price", fontsize = 16)
        plt.show()
        
        
        return np.sum(r,axis=1)
            
     
    def PlotPolicy(self):
        
        S = self.env.theta + 2*self.env.eta*torch.linspace(-1,1,51).float()
        
        for I in [-1, 0, 1]:
            
            X = torch.cat((S.reshape(-1,1), I * torch.ones((S.shape[0],1)) ), axis=1)
            Q = self.Q_main(X)
            
            a_idx = torch.argmax(Q, axis=1)
            
            a = self.action_idx_to_action(a_idx)
            
            plt.plot(S.detach().numpy(), a.detach().numpy(), label=r"$I=" + str(I) + "$", alpha=0.5, linewidth=(2-I))
            
        plt.legend()
        plt.show()