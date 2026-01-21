import torch
import torch.nn as nn

# __all__ = ['']

class ResidualMLPBlock(nn.Module):
    """
    Simple residual MLP block:
      y = x + Dropout(W2 * act(W1 * LN(x)))
    """
    def __init__(self, d_model: int, d_hidden: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        # x: [..., d_model]
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + self.drop(h)


class QuantileHead(nn.Module):
    """
    Residual block + final projection to quantiles.
    Input is a feature vector per example (flattened encoder output).
    """
    def __init__(self, in_dim: int, pred_len: int, n_pred_var: int, quantiles, dropout: float,
                 head_hidden: int = 512):
        super().__init__()
        self.pred_len = pred_len
        self.n_pred_var = n_pred_var
        self.quantiles = list(quantiles)
        self.n_q = len(self.quantiles)

        # Project to a compact head dimension, then residual refine, then project to outputs
        self.proj_in = nn.Linear(in_dim, head_hidden)
        self.resblock = ResidualMLPBlock(d_model=head_hidden, d_hidden=head_hidden * 4, dropout=dropout)
        self.proj_out = nn.Linear(head_hidden, pred_len * n_pred_var * self.n_q)

    def forward(self, features):
        # features: [B, in_dim]
        h = self.proj_in(features)                 # [B, head_hidden]
        h = self.resblock(h)                       # [B, head_hidden]
        out = self.proj_out(h)                     # [B, pred_len*n_pred_var*n_q]
        out = out.view(features.size(0), self.pred_len, self.n_pred_var, self.n_q)
        return out