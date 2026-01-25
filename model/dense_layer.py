
import torch.nn as nn

class Dense_Layer(nn.Module):
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