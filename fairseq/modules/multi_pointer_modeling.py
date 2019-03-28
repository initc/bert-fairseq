import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import math
import os
import sys
import numpy as np
import torch.nn as nn
from fairseq.modules.layer_norm import LayerNorm

from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


INF = 1e10
EPSILON = 1e-10

def positional_encodings_like(x, t=None):
    if t is None:
        positions = torch.arange(0., x.size(1))
        if x.is_cuda:
            positions = positions.cuda(x.get_device())
    else:
        positions = t
    encodings = torch.zeros(*x.size()[1:])
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())
    for channel in range(x.size(-1)):
        if channel % 2 == 0:
            encodings[:, channel] = torch.sin(
                positions / 10000 ** (channel / x.size(2)))
        else:
            encodings[:, channel] = torch.cos(
                positions / 10000 ** ((channel - 1) / x.size(2)))
    return Variable(encodings)


# LayerNorm = nn.LayerNorm


class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, dropout_ratio):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout_ratio)
        self.layernorm = LayerNorm(d_model)

    def forward(self, *x, padding=None):
        return self.layernorm(x[0] + self.dropout(self.layer(*x, padding=padding)))



# class MultiHead(nn.Module):

#     def __init__(self, d_key, d_value, n_heads, dropout_ratio, causal=False):
#         super().__init__()
#         self.attention = Attention(d_key, dropout_ratio, causal=causal)
#         self.wq = Linear(d_key, d_key, bias=False)
#         self.wk = Linear(d_key, d_key, bias=False)
#         self.wv = Linear(d_value, d_value, bias=False)
#         self.n_heads = n_heads

#     def forward(self, query, key, value, padding=None):
#         query, key, value = self.wq(query), self.wk(key), self.wv(value)
#         query, key, value = (
#             x.chunk(self.n_heads, -1) for x in (query, key, value))
#         return torch.cat([self.attention(q, k, v, padding=padding)
#                           for q, k, v in zip(query, key, value)], -1)



class LinearReLU(nn.Module):

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.feedforward = Feedforward(d_model, d_hidden, activation='relu')
        self.linear = nn.Linear(d_hidden, d_model)

    def forward(self, x, padding=None):
        return self.linear(self.feedforward(x))


# XD: adapted from fairseq transformer
class MultiHead(nn.Module):
    def __init__(self, d_key, d_value, n_heads, dropout_ratio, causal=False):
        super().__init__()
        assert d_key == d_value
        self.embed_dim = d_key
        self.num_heads = n_heads
        self.dropout = dropout_ratio
        self.head_dim = self.embed_dim // n_heads
        assert self.head_dim * n_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self._mask = None
        self.causal = causal  # XD

        self.in_proj_weight = nn.Parameter(torch.Tensor(3*self.embed_dim, self.embed_dim))

        bias = False
        if bias:
            self.in_proj_bias = nn.Parameter(torch.Tensor(3*self.embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        # self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

        self.reset_parameters()  # XD: initialized in BertWithMultiPointer.__init__

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            # nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, padding=None):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()
        assert qkv_same or kv_same  # XD
        
        if qkv_same:
            query = query.transpose(0, 1).contiguous() 
            key = query
            value = query
        elif kv_same:
            query = query.transpose(0, 1).contiguous() 
            key = key.transpose(0, 1).contiguous()
            value = key

        mask_future_timesteps = self.causal  # XD
        key_padding_mask = padding  # XD

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling
        src_len = k.size(0)
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        q = q.contiguous().view(tgt_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # only apply masking at training time (when incremental state is None)
        if mask_future_timesteps and tgt_len>1: # and incremental_state is None:
            assert query.size() == key.size(), \
                'mask_future_timesteps only applies to self-attention'
            attn_weights += self.buffered_mask(attn_weights).unsqueeze(0)
        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.float().masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = attn.transpose(0, 1)
        # attn = self.out_proj(attn)
        return attn

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2*self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2*self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        # XD: for in_proj_kv
        if False and self.shared_attn_module is not None and start == self.embed_dim and end is None:
            weight = self.shared_attn_module.in_proj_weight
            bias = self.shared_attn_module.in_proj_bias
        else:
            weight = self.in_proj_weight
            bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def buffered_mask(self, tensor):
        dim = tensor.size(-1)
        if self._mask is None:
            self._mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._mask.size(0) < dim:
            self._mask = torch.triu(fill_with_neg_inf(self._mask.resize_(dim, dim)), 1)
        return self._mask[:dim, :dim]

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, dimension, n_heads, hidden, dropout, causal=True):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead(dimension, dimension, n_heads,
                      dropout, causal),
            dimension, dropout)
        self.attention = ResidualBlock(
            MultiHead(dimension, dimension, n_heads,
                      dropout),
            dimension, dropout)
        self.feedforward = ResidualBlock(
            LinearReLU(dimension, hidden),
            dimension, dropout)

    def forward(self, x, encoding, context_padding=None, answer_padding=None):
        # print('encding size is: ', encoding.size())
        # exit(0)
        x = self.selfattn(x, x, x, padding=answer_padding)
        return self.feedforward(self.attention(x, encoding, encoding, padding=context_padding))


class TransformerDecoder(nn.Module): #only use the top layer's info of encoder.

    def __init__(self, dimension, n_heads, hidden, num_layers, dropout, causal=True):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(dimension, n_heads, hidden, dropout, causal=causal) for i in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.d_model = dimension

    def forward(self, x, encoding, context_padding=None, positional_encodings=True, answer_padding=None):
        if positional_encodings:
            x = x + positional_encodings_like(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoding, context_padding=context_padding, answer_padding=answer_padding)
        return x


class Linear(nn.Linear):

    def forward(self, x):
        size = x.size()
        return super().forward(
            x.contiguous().view(-1, size[-1])).view(*size[:-1], -1)


class Feedforward(nn.Module):

    def __init__(self, d_in, d_out, activation=None, bias=True, dropout=0.2):
        super().__init__()
        if activation is not None:
            self.activation = getattr(torch, activation)
        else:
            self.activation = lambda x: x
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.activation(self.linear(self.dropout(x)))

class DecoderAttention(nn.Module): #TJ
    def __init__(self, dim, dot=False):
        super().__init__()
        if not dot:
            self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(2 * dim, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None
        self.dot = dot

    def applyMasks(self, context_mask):
        self.context_mask = context_mask

    def forward(self, input, context, atten_mask):
        if not self.dot:
            targetT = self.linear_in(input)  # B x Ta x C
        else:
            targetT = input

        context_scores = torch.bmm(targetT, context.transpose(1, 2))

        context_scores.masked_fill_(atten_mask.unsqueeze(1), -float('inf'))
        context_weight = F.softmax(context_scores, dim=-1) + EPSILON
        context_atten = torch.bmm(context_weight, context)

        combined_representation = torch.cat([input, context_atten], 2)
        output = self.tanh(self.linear_out(combined_representation))

        return output, context_weight
