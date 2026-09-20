import torch.nn as nn 

class FFN(nn.Module):

    def __init__(self, d_model, hidden, dropout=0.1):
        super(FFN, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        return self.seq(x)