import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from .tools import Linear_fw, Conv2d_fw

class TextCNN(nn.Module):
    def __init__(self, input_dim, embd_size=128, in_channels=1, out_channels=128, kernel_heights=[3,4,5], dropout=0.5):
        super().__init__()
        '''
        cat((conv1-relu+conv2-relu+conv3-relu)+maxpool) + dropout, and to trans
        '''
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], input_dim), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], input_dim), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], input_dim), stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.embd = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(len(kernel_heights)*out_channels, embd_size)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2) # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def forward(self, frame_x):
        batch_size, seq_len, feat_dim = frame_x.size()
        frame_x = frame_x.view(batch_size, 1, seq_len, feat_dim)
        max_out1 = self.conv_block(frame_x, self.conv1)
        max_out2 = self.conv_block(frame_x, self.conv2)
        max_out3 = self.conv_block(frame_x, self.conv3)
        all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        fc_in = self.dropout(all_out)
        embd = self.embd(fc_in)
        return embd



class LSTMEncoder(nn.Module):
    ''' one directional LSTM encoder
    '''
    def __init__(self, input_size, hidden_size, embd_method='last'):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        assert embd_method in ['maxpool', 'attention', 'last']
        self.embd_method = embd_method
        
        if self.embd_method == 'attention':
            self.attention_vector_weight = nn.Parameter(torch.Tensor(hidden_size, 1))
            self.attention_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )
            self.softmax = nn.Softmax(dim=-1)

    def embd_attention(self, r_out, h_n):
        ''''
        paper: Hierarchical Attention Networks for Document Classification
        formulation:  lstm_output*softmax(u * tanh(W*lstm_output + Bias)
        W and Bias 是映射函数，其中 Bias 可加可不加
        u 是 attention vector 大小等于 hidden size
        '''
        hidden_reps = self.attention_layer(r_out)                       # [batch_size, seq_len, hidden_size]
        atten_weight = (hidden_reps @ self.attention_vector_weight)              # [batch_size, seq_len, 1]
        atten_weight = self.softmax(atten_weight)                       # [batch_size, seq_len, 1]
        # [batch_size, seq_len, hidden_size] * [batch_size, seq_len, 1]  =  [batch_size, seq_len, hidden_size]
        sentence_vector = torch.sum(r_out * atten_weight, dim=1)       # [batch_size, hidden_size]
        return sentence_vector

    def embd_maxpool(self, r_out, h_n):
        # embd = self.maxpool(r_out.transpose(1,2))   # r_out.size()=>[batch_size, seq_len, hidden_size]
                                                    # r_out.transpose(1, 2) => [batch_size, hidden_size, seq_len]
        in_feat = r_out.transpose(1,2)
        embd = F.max_pool1d(in_feat, in_feat.size(2), in_feat.size(2))
        return embd.squeeze(-1)

    def embd_last(self, r_out, h_n):
        #Just for  one layer and single direction
        return h_n.squeeze(0)

    def forward(self, x):
        '''
        r_out shape: seq_len, batch, num_directions * hidden_size
        hn and hc shape: num_layers * num_directions, batch, hidden_size
        '''
        r_out, (h_n, h_c) = self.rnn(x)
        embd = getattr(self, 'embd_'+self.embd_method)(r_out, h_n)
        return embd


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = Linear_fw(d_model, h * d_k)
        self.fc_k = Linear_fw(d_model, h * d_k)
        self.fc_v = Linear_fw(d_model, h * d_v)
        self.fc_o = Linear_fw(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            att = att.masked_fill(~attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class seqEncoder(nn.Module):
    def __init__(self, input_dim, embd_size=128, in_channels=1, kernel_heights=5, dropout=0.0) -> None:
        super().__init__()

        self.conv = Conv2d_fw(in_channels, embd_size, (kernel_heights, input_dim), padding=((kernel_heights-1)//2, 0))
        self.self_attn = ScaledDotProductAttention(d_model=embd_size, d_k=embd_size, d_v=embd_size, h=2, dropout=dropout)

    def embd_maxpool(self, r_out):

        in_feat = r_out.transpose(1,2)
        embd = F.max_pool1d(in_feat, in_feat.size(2), in_feat.size(2))
        return embd.squeeze(-1)


    def forward(self, x, mask=None):
        ''' x: modality sequences. [batch_size, seq_len, embd_size]'''

        b, l, d = x.size()
        x = x.view(b, 1, l, d)
        hidden = F.relu(self.conv(x).squeeze(3)).transpose(1,2)
        attn_hidden = self.self_attn(hidden, hidden, hidden, attention_mask=mask)
        return self.embd_maxpool(attn_hidden)
    

class seqOffset(nn.Module):
    def __init__(self, embd_size) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embd_size, int(embd_size/2))
        self.activation = nn.LeakyReLU()
        self.linear2 = nn.Linear(int(embd_size/2), embd_size)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.activation(out)
        return out + residual


if __name__ == '__main__':
    model = seqEncoder(5, 128, 1, 3, 0.5)
    input_x = torch.randn(32, 375, 5)
    output = model(input_x)
    print(output.shape)