import torch
import torch.nn as nn
import torch.nn.functional as F
from .self_attention import SelfAttention
from .layer_norm import LayerNorm

class MSVA(nn.Module):
    def __init__(self):
        super(MSVA, self).__init__()    
        self.cmb = [1,1,1]
        self.m = 365
        self.hidden = self.m
        self.apperture = 250
        self.att_1_3_size = 1024
        self.att1_3 = SelfAttention(apperture=self.apperture, input_size=self.att_1_3_size, output_size=self.att_1_3_size, dropout=0.5)
        self.ka1_3 = nn.Linear(in_features=self.att_1_3_size , out_features=365)
        self.kb = nn.Linear(in_features=self.ka1_3.out_features, out_features=365)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=512)
        self.kd = nn.Linear(in_features=self.kc.out_features, out_features=1)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout= nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=0)
        self.layer_norm_y_1_3 = LayerNorm(self.att_1_3_size)
        self.layer_norm_y_4 = LayerNorm(self.att_1_3_size)
        self.layer_norm_kc = LayerNorm(self.kc.out_features)

    def forward(self, x_list, seq_len):
        y_out_ls = []
        att_weights_ = []
        for i in range(len(x_list)):
            x = x_list[i].view(-1, x_list[i].shape[2])
            y, att_weights = self.att1_3(x)
            att_weights_  = att_weights
            y = y + x
            y = self.dropout(y)    
            y = self.layer_norm_y_1_3(y)
            y_out_ls.append(y)
        y_out_ls_filter = []
        for i in range(0,len(y_out_ls)):
            if(self.cmb[i]):
                y_out_ls_filter.append(y_out_ls[i])
        y_out = y_out_ls_filter[0]
        for i in range(1,len(y_out_ls)):
            y_out = y_out + y_out_ls_filter[i] 
        # Frame level importance score regression
        y = y_out
        y = self.ka1_3(y)
        y = self.kb(y)
        y = self.kc(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.layer_norm_kc(y)
        y = self.kd(y)
        y = self.sig(y)
        y = y.view(1, -1)
        return y, att_weights_