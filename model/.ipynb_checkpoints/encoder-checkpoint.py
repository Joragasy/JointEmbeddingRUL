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

from multihead_attention import MultiHeadSelfAttention
from positional_encoding import PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, ff_hidden, dropout=0.1):
        super().__init__()
        
        # Instantiate multi-head self-attention layer
        self.self_attn = MultiHeadSelfAttention(n_heads, d_model, dropout)
        
        # Define feedforward layer
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, d_model),
            nn.Dropout(dropout)
        )
        
        # Define layer normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Apply layer normalization to input
        x_norm = self.norm1(x)
        
        # Apply multi-head self-attention
        self_attn_output = self.self_attn(x_norm, mask=mask)
        
        # Add residual connection and apply layer normalization
        x1 = self.norm2(x + self_attn_output)
        
        # Apply feedforward layer
        ff_output = self.feedforward(x1)
        
        # Add residual connection
        output = x1 + ff_output
        
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, n_heads, d_model, ff_hidden, enc_seq_len, dropout=0.1):
        super().__init__()
        
        # Instantiate positional encoding
        self.pos_encoding = PositionalEncoding(d_model,max_seq_len=enc_seq_len)
        
        # Define list of encoder blocks
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(n_heads, d_model, ff_hidden, dropout) for _ in range(n_layers)
        ])
        
    def forward(self, x, mask=None):
        # Apply positional encoding to input
        
        x = self.pos_encoding(x)
        # Apply each encoder block in turn
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask=mask)
        return x