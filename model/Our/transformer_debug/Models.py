''' Define the Transformer model '''
from config import FLAGS
from transformer_debug.Layers import GraphEncoderLayer, CrossGraphEncoderLayer, PairDecoderLayer
import transformer_debug.Constants as Constants
import torch
import torch.nn as nn
import numpy as np

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            d_model=512, d_inner=2048, n_layers=6,
            n_head=8, d_k=64, d_v=64, dropout=0.1, n_outputs = 8):
        super().__init__()

        self.graph_encoder = GraphEncoder(
            d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.pair_decoder = PairDecoder(
            d_model=d_model, d_inner=d_model,
            n_layers=1, n_head=n_outputs, d_k=d_k, d_v=d_v,
            dropout=dropout)

        # self.tgt_node_prj_layers = nn.ModuleList()
        # D = d_model
        # while D > 1:
        #     D_next = D // down_proj_factor
        #     if D_next < 1:
        #         D_next = 1
        #     tgt_word_prj = nn.Linear(D, D_next, bias=False)  # TODO: may want bias, relu
        #     nn.init.xavier_normal_(tgt_word_prj.weight)
        #     self.tgt_node_prj_layers.append(tgt_word_prj)
        #     D = D_next

    def forward(self, g1_embs, g1_nids, g2_embs, g2_nids, src_nids_mask_g1=None):

        g1_embs, g2_embs, *_ = self.graph_encoder(g1_embs, g2_embs, g1_nids, g2_nids)

        sim_matrix = self.pair_decoder(g1_embs, g2_embs, g1_nids, g2_nids)

        '''
        map = self.map_encoder(g1_src_embs, g2_src_embs, src_nids)
        g1_debug = g1_embs.detach().cpu().numpy()
        g2_debug = g2_embs.detach().cpu().numpy()
        map = map.detach().cpu().numpy()
        seq_logit = map
        return seq_logit

        seq_logit = map
        # for tgt_word_prj in self.tgt_node_prj_layers:
        #     seq_logit = tgt_word_prj(seq_logit)
        # return seq_logit.view(-1, seq_logit.size(2))
        return seq_logit
        '''

        return sim_matrix

class GraphEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        graph_enc_full = []
        for _ in range(n_layers):
            graph_enc_full.append(CrossGraphEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout))
            graph_enc_full.append(GraphEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout))

        self.layer_stack = nn.ModuleList(graph_enc_full)

        self.MLP = nn.Linear(d_model,d_model)
        self.weights = nn.Parameter(torch.Tensor(1+2*n_layers).fill_(1.0))

        '''
        # No positional encoding for this model.
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(Constants.MAX_LEN, d_inner, padding_idx=0),
            freeze=True)
        '''

    def forward(self, g1_embs, g2_embs, g1_nids, g2_nids, return_attns=False):
        enc_slf_attn_list_intra, enc_slf_attn_list_inter = [], []

        # -- Prepare masks
        g1_attn_mask = get_attn_key_pad_mask(seq_k=g1_nids, seq_q=g1_nids)
        g2_attn_mask = get_attn_key_pad_mask(seq_k=g2_nids, seq_q=g2_nids)
        g1_g2_attn_mask = get_attn_key_pad_mask(seq_k=g1_nids, seq_q=g2_nids)
        g2_g1_attn_mask = get_attn_key_pad_mask(seq_k=g2_nids, seq_q=g1_nids)
        g1_non_pad_mask = get_non_pad_mask(g1_nids)
        g2_non_pad_mask = get_non_pad_mask(g2_nids)

        g1_out = g1_embs
        g2_out = g2_embs

        # -- Forward
        g1_merge = torch.unsqueeze(g1_embs, dim=0)
        g2_merge = torch.unsqueeze(g2_embs, dim=0)

        # intra-graph attention
        for i, enc_layer in enumerate(self.layer_stack):
            if i%2 == 0:
                g1_out, g2_out, g1_g2_attn = enc_layer(
                    g1_embs, g2_embs,
                    non_pad_mask=(g1_non_pad_mask, g2_non_pad_mask),
                    slf_attn_mask=(g2_g1_attn_mask, g1_g2_attn_mask))
                if return_attns:
                    enc_slf_attn_list_inter += [g1_g2_attn]
                g1_merge = torch.cat((g1_merge, torch.unsqueeze(g1_out, dim=0)), dim=0)
                g2_merge = torch.cat((g2_merge, torch.unsqueeze(g2_out, dim=0)), dim=0)
            else:
                g1_out, g2_out, g1_attn, g2_attn = enc_layer(
                    g1_out, g2_out,
                    non_pad_mask=(g1_non_pad_mask, g2_non_pad_mask),
                    slf_attn_mask=(g1_attn_mask, g2_attn_mask))
                if return_attns:
                    enc_slf_attn_list_intra += [(g1_attn, g2_attn)]
                g1_merge = torch.cat((g1_merge, torch.unsqueeze(g1_out, dim=0)), dim=0)
                g2_merge = torch.cat((g2_merge, torch.unsqueeze(g2_out, dim=0)), dim=0)

        #g1_out = self._combine_embeddings(g1_merge)
        #g2_out = self._combine_embeddings(g2_merge)

        if return_attns:
            return g1_out, g2_out, (enc_slf_attn_list_intra, enc_slf_attn_list_inter)

        return g1_out, g2_out,

    def _combine_embeddings(self, g_merge):
        g_merge = g_merge.to(FLAGS.device)
        g_merge = self.MLP(g_merge)
        g_merge = g_merge.permute(1,2,3,0) # B,N,D,K
        g_merge = self.weights*g_merge
        g_out = torch.sum(g_merge, dim=3)
        return g_out


class PairDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        assert n_layers == 1

        self.layer_stack = nn.ModuleList([PairDecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)])

    def forward(self, g1_embs, g2_embs, g1_nids, g2_nids):

        # -- Prepare masks
        g1_g2_attn_mask = get_attn_key_pad_mask(seq_k=g1_nids, seq_q=g2_nids)

        for dec_layer in self.layer_stack:
            g1_g2_attns = dec_layer(
                g1_embs, g2_embs,
                g1_g2_attn_mask=g1_g2_attn_mask)

        return g1_g2_attns,


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask
