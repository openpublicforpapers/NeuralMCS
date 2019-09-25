from config import FLAGS
from layers import create_act, NodeEmbedding, get_prev_layer
from utils_our import debug_tensor, pad_extra_rows
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ANPM(nn.Module):
    """ Attention_NTN_Padding_MNE layer. """

    def __init__(self, input_dim, att_times, att_num, att_style, att_weight,
                 feature_map_dim, bias, ntn_inneract, apply_u,
                 mne_inneract, mne_method, branch_style,
                 reduce_factor, criterion):
        super(ANPM, self).__init__()

        if 'an' in branch_style:
            self.att_layer = Attention(input_dim=input_dim,
                                       att_times=att_times,
                                       att_num=att_num,
                                       att_style=att_style,
                                       att_weight=att_weight)

            self.ntn_layer = NTN(
                input_dim=input_dim * att_num,
                feature_map_dim=feature_map_dim,
                inneract=ntn_inneract,
                apply_u=apply_u,
                bias=bias)
        else:
            raise ValueError('Must have the Attention-NTN branch')

        self.num_bins = 0
        if 'pm' in branch_style:
            self.mne_layer = MNE(
                input_dim=input_dim,
                inneract=mne_inneract)

            if 'hist_' in mne_method:  # e.g. hist_16_norm, hist_32_raw
                self.num_bins = self._get_mne_output_dim(mne_method)
            else:
                assert False

        self.mne_method = mne_method
        self.branch = branch_style

        proj_layers = []
        D = feature_map_dim + self.num_bins
        while D > 1:
            next_D = D // reduce_factor
            if next_D < 1:
                next_D = 1
            linear_layer = nn.Linear(D, next_D, bias=False)
            nn.init.xavier_normal_(linear_layer.weight)
            proj_layers.append(linear_layer)
            # if next_D != 1:
            #     proj_layers.append(nn.ReLU())
            D = next_D
        self.proj_layers = nn.ModuleList(proj_layers)

        if criterion == 'MSELoss':
            self.criterion = nn.MSELoss(reduction='mean')
        else:
            raise NotImplementedError()

    def forward(self, ins, batch_data, model):
        assert type(get_prev_layer(self, model)) is NodeEmbedding
        pair_list = batch_data.split_into_pair_list(ins, 'x')

        pairwise_embeddings = []
        true = torch.zeros(len(pair_list), 1, device=FLAGS.device)
        for i, pair in enumerate(pair_list):
            x1, x2 = pair.g1.x, pair.g2.x
            pairwise_embeddings.append(self._call_one_pair([x1, x2]))
            true[i] = pair.get_ds_true(
                FLAGS.ds_norm, FLAGS.dos_true, FLAGS.dos_pred, FLAGS.ds_kernel)

        # MLPs
        pairwise_embeddings = torch.cat(pairwise_embeddings, 0)
        for proj_layer in self.proj_layers:
            debug_tensor(pairwise_embeddings)
            pairwise_embeddings = proj_layer(pairwise_embeddings)
        pairwise_scores = pairwise_embeddings

        for pair in pair_list:
            pair.assign_ds_pred(pairwise_scores[i])

        assert pairwise_scores.shape == (len(pair_list), 1)
        loss = self.criterion(pairwise_scores, true)

        return loss

    def _call_one_pair(self, input):
        x_1 = input[0]
        x_2 = input[1]
        assert (x_1.shape[1] == x_2.shape[1])
        sim_scores, mne_features = self._get_ntn_scores_mne_features(x_1, x_2)

        # Merge.
        if self.branch == 'an':
            output = sim_scores
        elif self.branch == 'pm':
            output = mne_features
        elif self.branch == 'anpm':
            to_concat = (sim_scores, mne_features)
            output = torch.cat(to_concat, dim=1)
            # output = tf.nn.l2_normalize(output)
        else:
            raise RuntimeError('Unknown branching style {}'.format(self.branch))

        return output

    def _get_ntn_scores_mne_features(self, x_1, x_2):
        sim_scores, mne_features = None, None
        # Branch 1: Attention + NTN.
        if 'an' in self.branch:
            x_1_gemb = self.att_layer(x_1)
            x_2_gemb = self.att_layer(x_2)
            self.embeddings = [x_1_gemb, x_2_gemb]
            self.att = self.att_layer.att
            debug_tensor(self.att)
            debug_tensor(x_1_gemb)
            debug_tensor(x_2_gemb)
            sim_scores = self.ntn_layer([x_1_gemb, x_2_gemb])
            debug_tensor(sim_scores)
        # Branch 2: Padding + MNE.
        if 'pm' in self.branch:
            mne_features = self._get_mne_features(x_1, x_2)
        return sim_scores, mne_features

    def _get_mne_features(self, x_1, x_2):
        x_1_pad, x_2_pad = pad_extra_rows(x_1, x_2)
        mne_mat = self.mne_layer([x_1_pad, x_2_pad])
        if 'hist_' in self.mne_method:
            # mne_mat = mne_mat[:max_dim, :max_dim]
            with torch.no_grad():
                x = torch.histc(mne_mat.to('cpu'), bins=self.num_bins, min=0, max=1). \
                    view((1, -1)).to(FLAGS.device)  # TODO: check gradient
                # TODO: https://github.com/pytorch/pytorch/issues/1382; first to cpu then back
            x /= torch.sum(x)  # L1-normalize
        # elif self.mne_method == 'arg_max_naive':
        #     x_row = tf.reduce_max(mne_mat, reduction_indices=[0])
        #     x_col = tf.reduce_max(mne_mat, reduction_indices=[1])
        #     x = tf.reshape(tf.concat([x_row, x_col], 0), [1, -1])
        #     # sum over
        #     x = tf.reduce_sum(x)
        #     x = tf.reshape(x, [1, -1])
        else:
            raise ValueError('Unknown mne method {}'.format(self.mne_method))
        # output = tf.cast(x, tf.float32)
        # return output
        return x

    def _get_mne_output_dim(self, mne_method):
        if 'hist_' in mne_method:  # e.g. hist_16_norm, hist_32_raw
            ss = mne_method.split('_')
            assert (ss[0] == 'hist')
            rtn = int(ss[1])
        else:
            raise NotImplementedError()
        return rtn

class Attention(nn.Module):
    """ Attention layer."""

    def __init__(self, input_dim, att_times, att_num, att_style, att_weight):
        super(Attention, self).__init__()
        self.emb_dim = input_dim  # same dimension D as input embeddings
        self.att_times = att_times
        self.att_num = att_num
        self.att_style = att_style
        self.att_weight = att_weight
        assert (self.att_times >= 1)
        assert (self.att_num >= 1)
        assert (self.att_style == 'dot' or self.att_style == 'slm' or
                'ntn_' in self.att_style)

        self.vars = {}

        for i in range(self.att_num):
            self.vars['W_' + str(i)] = \
                glorot([self.emb_dim, self.emb_dim])
            if self.att_style == 'slm':
                self.interact_dim = 1
                self.vars['NTN_V_' + str(i)] = \
                    glorot([self.interact_dim, 2 * self.emb_dim])
            if 'ntn_' in self.att_style:
                self.interact_dim = int(self.att_style[4])
                self.vars['NTN_V_' + str(i)] = \
                    glorot([self.interact_dim, 2 * self.emb_dim])
                self.vars['NTN_W_' + str(i)] = \
                    glorot([self.interact_dim, self.emb_dim, self.emb_dim])
                self.vars['NTN_U_' + str(i)] = \
                    glorot([self.interact_dim, 1])
                self.vars['NTN_b_' + str(i)] = \
                    nn.Parameter([self.interact_dim])

    def forward(self, inputs):
        if type(inputs) is list:
            rtn = []
            for input in inputs:
                rtn.append(self._call_one_mat(input))
            return rtn
        else:
            return self._call_one_mat(inputs)

    def _call_one_mat(self, inputs):
        outputs = []
        for i in range(self.att_num):
            acts = [inputs]
            assert (self.att_times >= 1)
            output = None
            for _ in range(self.att_times):
                x = acts[-1]  # x is N*D
                temp = torch.mean(x, 0).view((1, -1))  # (1, D)
                h_avg = torch.tanh(torch.mm(temp, self.vars['W_' + str(i)])) if \
                    self.att_weight else temp
                self.att = self._gen_att(x, h_avg, i)
                output = torch.mm(self.att.view(1, -1), x)  # (1, D)
                x_new = torch.mul(x, self.att)
                acts.append(x_new)
            outputs.append(output)
        return torch.cat(outputs, 1)

    def _gen_att(self, x, h_avg, i):
        if self.att_style == 'dot':
            return interact_two_sets_of_vectors(
                x, h_avg, 1,  # interact only once
                W=[torch.eye(self.emb_dim, device=FLAGS.device)],
                act=torch.sigmoid)
        elif self.att_style == 'slm':
            # return tf.sigmoid(tf.matmul(concat, self.vars['a_' + str(i)]))
            return interact_two_sets_of_vectors(
                x, h_avg, self.interact_dim,
                V=self.vars['NTN_V_' + str(i)],
                act=torch.sigmoid)
        else:
            assert ('ntn_' in self.att_style)
            return interact_two_sets_of_vectors(
                x, h_avg, self.interact_dim,
                V=self.vars['NTN_V_' + str(i)],
                W=self.vars['NTN_W_' + str(i)],
                b=self.vars['NTN_b_' + str(i)],
                act=torch.sigmoid,
                U=self.vars['NTN_U_' + str(i)])


class NTN(nn.Module):
    """ NTN layer.
    (Socher, Richard, et al.
    "Reasoning with neural tensor networks for knowledge base completion."
    NIPS. 2013.). """

    def __init__(self, input_dim, feature_map_dim, apply_u,
                 inneract, bias):
        super(NTN, self).__init__()

        self.feature_map_dim = feature_map_dim
        self.apply_u = apply_u
        self.bias = bias
        self.inneract = create_act(inneract)

        self.vars = {}

        self.vars['V'] = glorot([feature_map_dim, input_dim * 2])
        self.vars['W'] = glorot([feature_map_dim, input_dim, input_dim])
        if self.bias:
            self.vars['b'] = nn.Parameter(torch.randn(feature_map_dim).to(FLAGS.device))
        if self.apply_u:
            self.vars['U'] = glorot([feature_map_dim, 1])

    def forward(self, inputs):
        assert len(inputs) == 2
        return self._call_one_pair(inputs)

    def _call_one_pair(self, input):
        x_1 = input[0]
        x_2 = input[1]

        # one pair comparison
        return interact_two_sets_of_vectors(
            x_1, x_2, self.feature_map_dim,
            V=self.vars['V'],
            W=self.vars['W'],
            b=self.vars['b'] if self.bias else None,
            act=self.inneract,
            U=self.vars['U'] if self.apply_u else None)


def interact_two_sets_of_vectors(x_1, x_2, interaction_dim, V=None,
                                 W=None, b=None, act=None, U=None):
    """
    Calculates the interaction scores between each row of x_1 (a marix)
        and x_2 ( a vector).
    :param x_1: an (N, D) matrix (if N == 1, a (1, D) vector)
    :param x_2: a (1, D) vector
    :param interaction_dim: number of interactions
    :param V:
    :param W:
    :param b:
    :param act:
    :param U:
    :return: if U is not None, interaction results of (N, interaction_dim)
             if U is None, interaction results of (N, 1)
    """
    feature_map = []
    for i in range(interaction_dim):
        middle = 0.0
        if V is not None:
            # In case x_2 is a vector but x_1 is a matrix, tile x_2.
            tiled_x_2 = torch.mul(torch.ones_like(x_1, device=FLAGS.device), x_2)
            concat = torch.cat((x_1, tiled_x_2), 1)
            v_weight = V[i].view(-1, 1)
            V_out = torch.mm(concat, v_weight)
            middle += V_out
        if W is not None:
            temp = torch.mm(x_1, W[i])
            h = torch.mm(temp, x_2.t())  # h = K.sum(temp*e2,axis=1)
            middle += h
        if b is not None:
            middle += b[i]
        feature_map.append(middle)

    output = torch.cat(feature_map, 1)

    output = F.normalize(output, p=1, dim=1)  # TODO: check why need this

    if act is not None:
        output = act(output)
    if U is not None:
        output = torch.mm(output, U)

    return output


class MNE(nn.Module):
    """ MNE layer. """

    def __init__(self, input_dim, inneract):
        super(MNE, self).__init__()

        self.inneract = create_act(inneract)
        self.input_dim = input_dim

    def forward(self, inputs):
        assert (type(inputs) is list and inputs)
        if type(inputs[0]) is list:
            # Double list.
            rtn = []
            for input in inputs:
                assert (len(input) == 2)
                rtn.append(self._call_one_pair(input))
            return rtn
        else:
            assert (len(inputs) == 2)
            return self._call_one_pair(inputs)

    def _call_one_pair(self, input):
        # Assume x_1 & x_2 are of dimension N * D
        x_1 = input[0]
        x_2 = input[1]

        # one pair comparison
        output = torch.mm(x_1, x_2.t())

        return self.inneract(output)


def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    rtn = nn.Parameter(torch.Tensor(*shape).to(FLAGS.device))
    nn.init.xavier_normal_(rtn)
    return rtn
