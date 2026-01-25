
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from tqdm import tqdm
import more_itertools as mit
import copy
import time
import random

random.seed(2)
torch.manual_seed(2)

from sklearn import preprocessing
import scipy.io as sio
from sklearn import linear_model
from scipy import interpolate
import csv


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        # Define linear layers for queries, keys, and values
        self.linear_q = nn.Linear(d_model, d_model).to(device , dtype=torch.float64)
        self.linear_k = nn.Linear(d_model, d_model).to(device , dtype=torch.float64)
        self.linear_v = nn.Linear(d_model, d_model).to(device , dtype=torch.float64)
        
        # Define output linear layer
        self.linear_out = nn.Linear(d_model, d_model)
        
        # Define dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Define scaling factor for dot product attention
        self.scale = torch.sqrt(torch.FloatTensor([d_model // n_heads])).to(device).to(torch.float64)
        
    def forward(self, x, encoder_output = None ,mask=None):
        batch_size = x.size(0)
        
        # Apply linear transformations to get queries, keys, and values
        if encoder_output is None:
            Q = self.linear_q(x)
            K = self.linear_k(x)
            V = self.linear_v(x)
        else:
            Q = self.linear_q(x)
            K = self.linear_k(encoder_output)
            V = self.linear_v(encoder_output)
        
        # Split the heads
        Q = Q.view(batch_size, -1, self.n_heads, self.d_model // self.n_heads).transpose(1,2)
        K = K.view(batch_size, -1, self.n_heads, self.d_model // self.n_heads).transpose(1,2)
        V = V.view(batch_size, -1, self.n_heads, self.d_model // self.n_heads).transpose(1,2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax activation
        attention = torch.softmax(scores, dim=-1)
        
        # Apply dropout
        attention = self.dropout(attention)
        
        # Compute weighted sum of values
        context = torch.matmul(attention, V)
        
        # Concatenate and reshape the heads
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.n_heads * (self.d_model // self.n_heads))
        
        # Apply output linear layer
        output = self.linear_out(context)
        
        return output