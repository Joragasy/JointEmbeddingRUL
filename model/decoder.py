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

from .multihead_attention import MultiHeadSelfAttention
from .positional_encoding import PositionalEncoding

device = "cuda" if torch.cuda.is_available() else "cpu"

class DecoderLayer(torch.nn.Module):
    def __init__(self, n_heads, d_model,dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadSelfAttention(n_heads, d_model, dropout=dropout).to(device , dtype=torch.float64)
        self.attn2 = MultiHeadSelfAttention(n_heads, d_model, dropout=dropout).to(device , dtype=torch.float64)
        self.fc1 = nn.Linear(d_model, d_model).to(device , dtype=torch.float64)
        self.fc2 = nn.Linear(d_model, d_model).to(device , dtype=torch.float64)
        self.dropout = nn.Dropout(dropout).to(device , dtype=torch.float64)
        self.norm1 = nn.LayerNorm(d_model).to(device , dtype=torch.float64)
        self.norm2 = nn.LayerNorm(d_model).to(device , dtype=torch.float64)
        self.norm3 = nn.LayerNorm(d_model).to(device , dtype=torch.float64)
        
    def forward(self, x, enc_output=None , mask=None):
        x = x.to(device , dtype=torch.float64)
        if enc_output is not None :
            enc_output = enc_output.to(device , dtype=torch.float64)
        a = self.attn1(x) 
        x = self.norm1(a + x)        
        a = self.attn2(x, encoder_output = enc_output , mask=mask)
        x = self.norm2(a + x)
        a = self.fc1(F.elu(self.fc2(x)))       
        x = self.norm3(x + a)
        return x 

class TransformerDecoder(torch.nn.Module):
    #new params :  n_decoder_layers, n_heads , n_features, dec_seq_len,output_sequence_length
    # dim_val, dim_attn, useful_sensors_number, dec_seq_len,output_sequence_length, n_features,  n_heads , n_decoder_layers
    def __init__(self, n_decoder_layers, n_heads, n_features, d_model, dec_seq_len, out_seq_len, dropout = 0.1):
        
        super(TransformerDecoder, self).__init__()

        self.dec_seq_len = dec_seq_len
        self.pos_encoding = PositionalEncoding(n_features,max_seq_len=dec_seq_len)
        #Initiate Decoder
        self.decoder = []
        for i in range(n_decoder_layers):
            self.decoder.append(DecoderLayer(n_heads, d_model))
        
        self.dec_input_fc = nn.Linear(n_features, d_model)
    
    def forward(self,x, features_fusion=None):
        #decoder receive the output of feature fusion layer.
        x = x[:,-self.dec_seq_len:]
        x = x.to(device , dtype=torch.float64)
        # Positional Encoding
        x = self.pos_encoding(x)
        features_fusion = features_fusion.to(device , dtype=torch.float64)
        d = self.decoder[0](self.dec_input_fc(x), features_fusion)   
        if len(self.decoder) > 1 :
            for n in range(1,len(self.decoder)):
                d = self.decoder[n](d)
        #x = self.out_fc(d.flatten(start_dim=1))     
        return d

class Dense_Layer(torch.nn.Module):
    def __init__(self, dim_val, dec_seq_len, dropout = 0.1):
        super(Dense_Layer,self).__init__()
        self.fc = nn.Sequential ( 
            nn.Linear(dec_seq_len * dim_val, 64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self,x):
        x = x.flatten(start_dim=1)
        out = self.fc(x)
        return out