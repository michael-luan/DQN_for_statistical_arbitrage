# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:35:23 2022

@author: sebja
"""

from Environment import Environment
from DDQN import DDQN
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt

import copy

# define the base set of parameters
S0 = 1  
theta= 1
kappa = 0.5
eta = 0.02
beta= 0.01
delta = 0.005
Q = 10
T = 100

# Define the state space and action space
state_size = 3 # current price, current inventory, time remaining
action_size = 21 # integers from -Q to Q


np.random.seed(123321)
env = Environment(S0, theta, kappa, eta, beta, delta, Q, T)
ddqn = DDQN(env, gamma=1, lr=1e-3)
ddqn.env = env
r = ddqn.Train(n_iter = 4_001, mini_batch_size=256, n_plot= 1000)




#%%

kappas = [0.2, 0.5, 0.9]
c = ["red", "green", "blue"]
hists = []
for kappa in kappas:
    if kappa == kappas[0]:
        np.random.seed(3231413)
    elif kappa == kappas[1]:
        np.random.seed(3231412)
    else:
        np.random.seed(4)
    env = Environment(S0, theta, kappa, eta, beta, delta, Q, T)
    ddqn = DDQN(env, gamma=1, lr=1e-3)
    ddqn.env = env
    temp = ddqn.Train(n_iter = 2001, mini_batch_size=1000, n_plot= 2000)
    hists.append(temp)
    
for i in range(3):
    plt.hist(hists[i], bins=40, density=True, label = r"$\kappa = $" + str(kappas[i]), color=c[i], alpha = 0.3)
    plt.xlabel("Expected Sum of Rewards")
    plt.ylabel("Density")
    plt.legend()
    plt.title(r"Expected Sum of Rewards Under Various $\kappa$", fontsize = 16)
    
#%%

betas = [0.005, 0.01, 0.015]
S0 = 1
theta= 1
kappa = 0.5
eta = 0.02
delta = 0.005
Q = 10
T = 10
hists = []
for beta in betas:
    if beta == betas[0]:
        np.random.seed(3231413)
    elif beta == betas[1]:
        np.random.seed(3231413)
    else:
        np.random.seed(3231413)
    env = Environment(S0, theta, kappa, eta, beta, delta, Q, T)
    ddqn = DDQN(env, gamma=1, lr=1e-3)
    ddqn.env = env
    temp = ddqn.Train(n_iter = 2001, mini_batch_size=1000, n_plot= 2000)
    hists.append(temp)
    
for i in range(3):
    plt.hist(hists[i], bins=40, density=True, label = r"$\beta = $" + str(betas[i]), color=c[i], alpha = 0.3)
    plt.xlabel("Expected Sum of Rewards")
    plt.ylabel("Density")
    plt.legend()
    plt.title(r"Expected Sum of Rewards Under Various $\beta$", fontsize = 16)
    
#%%
c = ["red", "green", "blue"]
etas = [0.015, 0.02, 0.025]
S0 = 1
theta= 1
kappa = 0.5
beta = 0.01
delta = 0.005
Q = 10
T = 10
hists = []
for eta in etas:
    if eta == etas[0]:
        np.random.seed(32314123)
    elif eta == etas[1]:
        np.random.seed(32314)
    else:
        np.random.seed(3231413)
    env = Environment(S0, theta, kappa, eta, beta, delta, Q, T)
    ddqn = DDQN(env, gamma=1, lr=1e-3)
    ddqn.env = env
    temp = ddqn.Train(n_iter = 2001, mini_batch_size=1000, n_plot=2000)
    hists.append(temp)
    
#%%    

for i in range(3):
    # plt.hist(hists[i], bins=40, density=True, label = r"$\eta = $" + str(etas[i]), color=c[i], alpha = 0.3)
    sns.kdeplot(hists[i], label = r"$\eta = $" + str(etas[i]), color=c[i])
    plt.xlabel("Expected Sum of Rewards")
    plt.ylabel("Density")
    plt.legend()
    plt.title(r"Expected Sum of Rewards Under Various $\eta$", fontsize = 16)