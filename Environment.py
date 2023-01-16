# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:23:45 2022

@author: sebja
"""

import numpy as np
import torch

class Environment():
    
    def __init__(self, S0, theta, kappa, eta, beta, delta, Q, T):
        
        self.S0 = S0
        self.theta = theta
        self.kappa = kappa
        self.eta = eta
        self.beta = beta
        self.delta = delta
        self.Q = Q
        self.T = T
        self.action_space = np.arange(-Q, Q, step = 1  )
        
        
    def randomise_S0(self, size=1):
        
        return self.theta + self.eta * np.random.randn(size)
        
    def step(self, S, I, T, q, nsims = 1):
        """
        Takes a step in the environment with a given action.
        """
        
        # modify price evolution based on action
        S_new = self.theta + (S - self.theta) * np.exp(-self.kappa) \
            + self.beta * np.sign(q - I) * np.sqrt(np.abs(q - I)) \
                + self.eta * torch.randn(nsims)
        
        S_new = S_new.float()
        
        # update time remaining
        T -= 1
        
        # compute reward
        r = q * (S_new - S) - ((q - I) * S + self.delta * np.abs(q - I))
        
        # update inventory
        I = q
        
        
        
        return S_new, I, T, r
    