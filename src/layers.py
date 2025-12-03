import torch 
import torch.nn as nn
import torch.nn.functional as F

import math


class LayerNorm(nn.Module):

    def __init__(self, input_size, eps=1e-9):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))
        self.eps = eps

    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True, unbiased=False)


        o = (x - x_mean) / torch.sqrt(x_var + self.eps)
        o = o*self.gamma + self.beta

        return o
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, sequence_length=200):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(sequence_length, d_model)
        pos = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        exponent = torch.arange(0, d_model, 2, dtype=torch.float) / d_model
        div_term = torch.pow(10000, exponent)

        result = pos/div_term
        pe[:, 0::2] = torch.sin(result)
        pe[:, 1::2] = torch.cos(result)
        self.register_buffer('pe', pe.unsqueeze(0))


    def forward(self, x):
        o = x + self.pe[:,:x.shape[1],:]
        return o
    

class FeedForwardLayer(nn.Module):

    def __init__(self, input_size, output_size):
        super(FeedForwardLayer, self).__init__()
        hidden_size = 2048

        self.sequence = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.sequence(x)


class MultiHeadAttention(nn.Module):

    def __init__(self, head_cnt, d_model):
        super(MultiHeadAttention, self).__init__()
        self.head_cnt = head_cnt
        self.d_model = d_model
        assert d_model % self.head_cnt == 0, "d_model whould be divisible by head_cnt."
        self.head_size = d_model // self.head_cnt
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, key, value, mask=None, q_mask=None):

        # Create one "large head" 
        Q_proj = self.Q(query)
        K_proj = self.K(key)
        V_proj = self.V(value)

        # The key and value vectors are expected to be the same length. 
        assert key.shape[1] == value.shape[1]

        # Create views where each view represents a single head. 
        batch_size, _, _ = query.shape
        Q_proj = Q_proj.view(query.shape[0], query.shape[1], self.head_cnt, self.head_size).transpose(1, 2)
        K_proj = K_proj.view(key.shape[0], key.shape[1], self.head_cnt, self.head_size).transpose(1, 2)
        V_proj = V_proj.view(value.shape[0], value.shape[1], self.head_cnt, self.head_size).transpose(1, 2)

        # Do attention operation on each view. 
        scores = Q_proj @ K_proj.transpose(-2, -1) / math.sqrt(self.head_size)
        
        # Add mask if necessary. 
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1) 
        o = attention_weights @ V_proj 

        # concat back all the attention heads. 
        o = o.transpose(1, 2).contiguous()
        o = o.view(batch_size, query.shape[1], -1)

        if q_mask is not None:
            o = o.masked_fill(q_mask, 0.0)

        # complete final projection. This mixes the information of all the heads together.         
        o_proj = self.O(o)
        return o_proj, scores


class EncoderLayer(nn.Module):

    def __init__(self, model_size=512, head_cnt=8, dropout=0.1):

        super(EncoderLayer, self).__init__()
        self.model_size = model_size
        self.head_cnt = head_cnt

        # MHA sublayer
        self.attention_layer = MultiHeadAttention(self.head_cnt, self.model_size)
        self.norm1 = LayerNorm(self.model_size)
        self.dropout1 = nn.Dropout(dropout)

        # FFN Sublayer 
        self.feed_forward_layer = FeedForwardLayer(self.model_size, self.model_size)
        self.norm2 = LayerNorm(self.model_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        # Run prenorm -> MHA -> dropout -> residual 
        x_norm1 = self.norm1(x) 
        attention_layer_output, _ = self.attention_layer(x_norm1, x_norm1, x_norm1, attention_mask)
        x_residual1 = x + self.dropout1(attention_layer_output)

        # Run prenorm -> FFN -> dropout -> residual 
        x_norm2 = self.norm2(x_residual1)
        ffn_output = self.feed_forward_layer(x_norm2)
        x_dropout2 = x_residual1 + self.dropout2(ffn_output)

        return x_dropout2


class Encoder(nn.Module): 

    def __init__(self, vocab_size, model_size=512, head_cnt=8, stack_cnt=4):
        super().__init__()

        self.dropout=0.1
        self.embedding = nn.Embedding(vocab_size, model_size)
        self.positional_encoder = PositionalEncoding(model_size)
        self.dropout = nn.Dropout(self.dropout)
        self.encoders = nn.ModuleList(
            [
                EncoderLayer(model_size, head_cnt) for _ in range(stack_cnt)
            ]
        )
        self.norm = LayerNorm(model_size)
        

    def forward(self, x):

        attention_mask = self.create_encoder_mask(x)
        x_embeddings = self.embedding(x)
        x_encoded_embeddings = self.positional_encoder(x_embeddings)
        x_encoded_embeddings = self.dropout(x_encoded_embeddings)

        o = x_encoded_embeddings
        for layer in self.encoders:
            o = layer(o, attention_mask)

        o_norm = self.norm(o)
        return o_norm 
    
    def create_encoder_mask(self, x):
        attention_mask = (x == 0).unsqueeze(1).unsqueeze(1)
        return attention_mask
    

class DecoderLayer(nn.Module): 

    def __init__(self, model_size=512, head_cnt=8, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.model_size = model_size
        self.head_cnt = head_cnt

        # self-attention MHA layer. 
        self.norm1 = LayerNorm(self.model_size)
        self.masked_attention_layer = MultiHeadAttention(self.head_cnt, self.model_size)        
        self.dropout1 = nn.Dropout(dropout)

        # cross-attention MHA layer 
        self.norm2 = LayerNorm(self.model_size)
        self.cross_attention_layer = MultiHeadAttention(self.head_cnt, self.model_size)
        self.dropout2 = nn.Dropout(dropout)

        # FFN layer 
        self.norm3 = LayerNorm(self.model_size)
        self.feed_forward_layer = FeedForwardLayer(self.model_size, self.model_size)
        self.dropout3 = nn.Dropout(dropout)


    def forward(self, query, key, value, self_attention_mask=None, cross_attention_mask=None):

        # self-attention MHA layer 
        q_norm = self.norm1(query)
        self_attention_output, _ = self.masked_attention_layer(q_norm, q_norm, q_norm, self_attention_mask)
        self_mha_output_norm = query + self.dropout1(self_attention_output)

        # cross-attention MHA layer 
        self_attention_norm = self.norm2(self_mha_output_norm)
        o_proj2, _ = self.cross_attention_layer(self_attention_norm, key, value, cross_attention_mask)
        cross_mha_output_norm = self_mha_output_norm + self.dropout2(o_proj2)

        # Do FFN layer 
        cross_attention_norm = self.norm3(cross_mha_output_norm)
        o = self.feed_forward_layer(cross_attention_norm)
        ffn_output = cross_mha_output_norm + self.dropout3(o)

        return ffn_output


class Decoder(nn.Module):

    def __init__(self, vocab_size, model_size=512, head_cnt=8, stack_cnt=4):
        super(Decoder, self).__init__()

        # Setup embeddings and positional encodings. 
        self.embedding = nn.Embedding(vocab_size, model_size)
        self.positional_encoder = PositionalEncoding(model_size)
        self.dropout = nn.Dropout(0.1)
        
        # Setup decoder layers
        self.decoders = nn.ModuleList(
            [DecoderLayer(model_size, head_cnt) for _ in range(stack_cnt)]
        )

        # Setup final layernorm. 
        self.norm = LayerNorm(model_size)

    def forward(self, query, key, value, encoder_attention_mask=None):
        # Do embedding and positional encoding. 
        q_embeddings = self.embedding(query)
        q_encoded_embeddings = self.positional_encoder(q_embeddings)
        q_encoded_embeddings = self.dropout(q_encoded_embeddings)

        # Create self-attention mask 
        self_attention_mask = self.create_self_attention_mask(query)

        # Create encoder mask. 
        o = q_encoded_embeddings
        for layer in self.decoders:
            o = layer(o, key, value, self_attention_mask, encoder_attention_mask)

        o_norm = self.norm(o)
        return o_norm
    

    def create_self_attention_mask(self, x):
    
        # Create causality mask. to prevent leftward information flow. 
        batch_size, seq_len = x.shape
        upper_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        causality_mask = upper_mask.unsqueeze(0).unsqueeze(0)
        # We want dims 1 x 1 x seq_len x seq_len 
        assert causality_mask.shape == (1, 1, seq_len, seq_len)

        # Create padding mask 
        # 2. Padding Mask (for the target sequence itself)
        # (Batch, 1, 1, Seq_Len)
        padding_mask = (x == 0).unsqueeze(1).unsqueeze(2)
        
        # We unsqueeze for the head dimension. 
        self_attention_mask = (causality_mask | padding_mask)

        assert self_attention_mask.shape == (batch_size, 1, seq_len, seq_len), \
            "The decoder's self attention mask has the correct shape."
        return self_attention_mask
