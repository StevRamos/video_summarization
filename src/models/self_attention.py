import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, apperture, ignore_itself=False, input_size=1024, output_size=1024,dropout=0.5): #apperture -1 to ignore
        super(SelfAttention, self).__init__()
        self.apperture = apperture
        self.ignore_itself = ignore_itself
        self.m = input_size
        self.output_size = output_size
        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        n = x.shape[0]
        K = self.K(x)  
        Q = self.Q(x)  
        V = self.V(x)
        Q *= 0.06
        logits = torch.matmul(Q, K.transpose(1,0))
        if self.ignore_itself:
            logits[torch.eye(n).byte()] = -float("Inf")
        if self.apperture > 0:
            onesmask = torch.ones(n, n)
            trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
            logits[trimask == 1] = -float("Inf")
        att_weights_ = nn.functional.softmax(logits, dim=-1)
        weights = self.dropout(att_weights_)
        y = torch.matmul(V.transpose(1,0), weights).transpose(1,0)
        y = self.output_linear(y)
        return y, att_weights_