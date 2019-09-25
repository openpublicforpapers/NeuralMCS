''' Define the Layers '''
from transformer_debug.SubLayers import MultiHeadAttention, MultiHeadCrossGraphAttention, PositionwiseFeedForward
import torch.nn as nn

class GraphEncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(GraphEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, g1_out, g2_out, non_pad_mask=(None,None), slf_attn_mask=(None,None)):
        g1_out, g1_attn = self.slf_attn(
            g1_out, g1_out, g1_out, mask=slf_attn_mask[0])

        g2_out, g2_attn = self.slf_attn(
            g2_out, g2_out, g2_out, mask=slf_attn_mask[1])

        g1_out *= non_pad_mask[0]
        g1_out = self.pos_ffn(g1_out)
        g1_out *= non_pad_mask[0]

        g2_out *= non_pad_mask[1]
        g2_out = self.pos_ffn(g2_out)
        g2_out *= non_pad_mask[1]

        return g1_out, g2_out, g1_attn, g2_attn

class CrossGraphEncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(CrossGraphEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadCrossGraphAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, g1_out, g2_out, non_pad_mask=(None, None), slf_attn_mask=(None, None)):
        g1_out, g2_out, g1_g2_attn = self.slf_attn(
            g1_out, g2_out, mask=slf_attn_mask)

        g1_out *= non_pad_mask[0]
        g1_out = self.pos_ffn(g1_out)
        g1_out *= non_pad_mask[0]

        g2_out *= non_pad_mask[1]
        g2_out = self.pos_ffn(g2_out)
        g2_out *= non_pad_mask[1]

        return g1_out, g2_out, g1_g2_attn

class PairDecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(PairDecoderLayer, self).__init__()
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

    def forward(self, g1_embs, g2_embs, g1_g2_attn_mask):
        # TODO: we're doing extra computations here
        _, g1_g2_attn = self.enc_attn(g2_embs, g1_embs, g1_embs, mask=g1_g2_attn_mask)
        return g1_g2_attn
