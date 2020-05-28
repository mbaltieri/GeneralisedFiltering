# This module containes the definition of a generic probabilistc model

import numpy as np
import torch
import functions

# class HDM():
#     def __init__(self, l, m=1, n=0, r=0, p=0, h=0, e_n=0, e_n=0, e_r=0, e_p=0, e_h=0):
#         self.l = l                               # number of layers

#         self.m = m                               # observations dimension
#         self.n = n                               # hidden states dimension
#         self.r = r                               # inputs dimension
#         self.p = p                               # parameters dimension
#         self.h = h                               # hyperparameters dimension

#         self.e_n = e_n                           # embedding dimension observations
#         self.e_n = e_n                           # embedding dimension hidden states
#         self.e_r = e_r                           # embedding dimension inputs
#         self.e_p = e_p                           # embedding dimension parameters
#         self.e_h = e_h                           # embedding dimension hyperparameters

#         for i in range(self.l):
#             self.layers[i] = layer()            # create layers

#     def addLayer(self, llayer, level):
#         self.layers[level] = llayer

#     # FIXME: check for boundary conditions (first and last layer)
#     # FIXME: make sure that 'self.layers' is an array of objects
#     # FIXME: check the arguments for 'forward' in 'layers' class
#     def forward(self):
#         for i in range(self.l):
#             output = self.layers[i](output)
#         return output

#     # FIXME: errors per layer, add and use layer argument
#     def epsilon_z(self, layer):
#         return self.y[layer,:,:] - self.g(self)
    
#     # TODO: check how weighted prediction errors are defined
#     def xi_z(self, layer):
#         return torch.bmm(Pi_z, epsilon_z(self))