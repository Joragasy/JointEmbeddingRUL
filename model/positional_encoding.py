import torch
import torch.nn as nn
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=60):
        super().__init__()
        
        # Compute the positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model).to(device)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register the positional encoding matrix as a buffer
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add the positional encoding to the input tensor
        #print("size of PosEnc input {} , size of PosEnc {}".format(x.shape, self.pe.shape) )
        #print(x.shape , self.pe.shape)
        x = x + self.pe[:, :x.size(1), :]
        return x