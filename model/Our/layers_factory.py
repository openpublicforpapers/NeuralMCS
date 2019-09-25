from config import FLAGS
from layers import NodeEmbedding, NodeEmbeddingCombinator, \
    NodeEmbeddingInteraction, Sequence, Fancy, Loss, MatchingMatrixComp
from layers_gmn_cvpr import Affinity, PowerIteration, BiStochastic
from layers_pca import PCAModel
from layers_transformer_debug import OurTransformerPrototype
from layers_gmn_icml import GMNPropagator, GMNAggregator, GMNLoss, MLP
from layers_redacted_1 import ANPM
from layers_redacted_2 import GraphConvolutionCollector, CNN
from layers_our_loss import OurLossFunction
from layers_tree_search import TreeSearch
import torch.nn as nn


def create_layers(model, pattern, num_layers):
    layers = nn.ModuleList()
    for i in range(1, num_layers + 1):  # 1-indexed
        sp = vars(FLAGS)['{}_{}'.format(pattern, i)].split(':')
        name = sp[0]
        layer_info = {}
        if len(sp) > 1:
            assert (len(sp) == 2)
            for spec in sp[1].split(','):
                ssp = spec.split('=')
                layer_info[ssp[0]] = '='.join(ssp[1:])  # could have '=' in layer_info
        if name in layer_ctors:
            layers.append(layer_ctors[name](layer_info, model, i, layers))
        else:
            raise ValueError('Unknown layer {}'.format(name))
    return layers


def create_NodeEmbedding_layer(lf, model, layer_id, *unused):
    _check_spec([4, 5], lf, 'NodeEmbedding')
    input_dim = lf.get('input_dim')
    if not input_dim:
        if layer_id != 1:
            raise RuntimeError(
                'The input dim for layer must be specified'.format(layer_id))
        input_dim = model.num_node_feat
    else:
        input_dim = int(input_dim)
    return NodeEmbedding(
        type=lf['type'],
        in_dim=input_dim,
        out_dim=int(lf['output_dim']),
        act=lf['act'],
        bn=_parse_as_bool(lf['bn']))


def create_NodeEmbeddingInteraction_layer(lf, *unused):
    _check_spec([2], lf, 'NodeEmbeddingInteraction')
    return NodeEmbeddingInteraction(
        type=lf['type'],
        in_dim=int(lf['input_dim']))


def create_NodeEmbeddingCombinator_layer(lf, model, layer_id, layers):
    _check_spec([2], lf, 'NodeEmbeddingCombinator')
    return NodeEmbeddingCombinator(
        from_layers=_parse_as_int_list(lf['from_layers']),
        layers=layers,
        num_node_feat=model.num_node_feat,
        style=lf['style'])


def create_Affinity_layer(lf, *unused):
    _check_spec([2], lf, 'Affinity')
    return Affinity(
        F=int(lf['F']),
        U=int(lf['U']))


def create_PowerIteration_layer(lf, *unused):
    _check_spec([1], lf, 'PowerIteration')
    return PowerIteration(k=int(lf['k']))


def create_BiStochastic_layer(lf, *unused):
    _check_spec([1], lf, 'BiStochastic')
    return BiStochastic(k=int(lf['k']))


def create_Sequence_layer(lf, *unused):
    _check_spec([2], lf, 'Sequence')
    return Sequence(
        type=lf['type'],
        in_dim=int(lf['input_dim']))


def create_OurTransformerPrototype_layer(lf, model, layer_id, layers):
    _check_spec([10], lf, 'OurTransformerPrototype')
    prev_layer = layers[layer_id - 2]  # 1-based to 0-based --> prev
    in_dim = int(lf['input_dim'])
    d_model = int(lf['d_model'])
    d_inner = int(lf['d_inner'])
    d_k = int(lf['d_k'])
    d_v = int(lf['d_v'])
    if type(prev_layer) is NodeEmbeddingCombinator:
        d = prev_layer.out_dim
        assert d >= in_dim
        delta_d = d - in_dim
        in_dim += delta_d  # d
        d_model += delta_d
        d_inner += delta_d
        d_k += delta_d
        d_v += delta_d
    return OurTransformerPrototype(
        in_dim=in_dim,
        d_model=d_model, d_inner=d_inner,
        n_layers=int(lf['n_layers']), n_head=int(lf['n_head']),
        d_k=d_k, d_v=d_v,
        dropout=float(lf['dropout']),
        beam_size=int(lf['beam_size']),
        n_outputs=int(lf['n_outputs']))


def create_Fancy_layer(lf, *unused):
    _check_spec([1], lf, 'Fancy')
    return Fancy(in_dim=int(lf['input_dim']))


def create_Loss_layer(lf, *unused):
    _check_spec([1], lf, 'Loss')
    return Loss(
        type=lf['type'])


def create_GMNEncoder_layer(lf, model, layer_id, *unused):
    _check_spec([2, 3, 4], lf, 'GMNEncoder')
    if layer_id == 1:
        input_dim = model.num_node_feat
    else:
        input_dim = int(lf['input_dim'])
    return MLP(
        input_dim=input_dim,
        output_dim=int(lf['output_dim']),
        activation_type=lf['act']
    )


def create_GMNPropagator_layer(lf, model, layer_id, *unused):
    _check_spec([2, 3, 4], lf, 'GMNPropagator')

    f_node = lf.get('f_node')
    if not f_node:
        f_node = 'MLP'
    return GMNPropagator(
        input_dim=int(lf['input_dim']),
        output_dim=int(lf['output_dim']),
        f_node=f_node
    )


def create_GMNAggregator_layer(lf, *unused):
    _check_spec([2], lf, 'GMNAggregator')
    return GMNAggregator(
        input_dim=int(lf['input_dim']),
        output_dim=int(lf['output_dim'])
    )


def create_GMNLoss_layer(lf, *unused):
    _check_spec([0, 1], lf, 'GMNLoss')
    return GMNLoss(ds_metric=lf['ds_metric'])


def create_ANPM_layer(lf, *unused):
    _check_spec([15], lf, 'ANPM')
    return ANPM(
        input_dim=int(lf['input_dim']),
        att_times=int(lf['att_times']),
        att_num=int(lf['att_num']),
        att_style=lf['att_style'],
        att_weight=_parse_as_bool(lf['att_weight']),
        feature_map_dim=int(lf['feature_map_dim']),
        bias=_parse_as_bool(lf['bias']),
        ntn_inneract=lf['ntn_inneract'],
        apply_u=_parse_as_bool(lf['apply_u']),
        mne_inneract=lf['mne_inneract'],
        mne_method=lf['mne_method'],
        branch_style=lf['branch_style'],
        reduce_factor=int(lf['reduce_factor']),
        criterion=lf['criterion'])


def create_GraphConvolutionCollector_layer(lf, *unused):
    _check_spec([5], lf, 'GraphConvolutionCollector')
    return GraphConvolutionCollector(
        gcn_num=int(lf['gcn_num']),
        fix_size=int(lf['fix_size']),
        mode=int(lf["mode"]),
        padding_value=int(lf["padding_value"]),
        align_corners=_parse_as_bool(lf["align_corners"])
    )


def create_CNN_layer(lf, *unused):
    _check_spec([9], lf, 'CNN')
    return CNN(
        in_channels=int(lf['in_channels']),
        out_channels=int(lf['out_channels']),
        kernel_size=int(lf["kernel_size"]),
        stride=int(lf["stride"]),
        gcn_num=int(lf["gcn_num"]),
        bias=_parse_as_bool(lf['bias']),
        poolsize=int(lf["poolsize"]),
        act=lf['act'],
        end_cnn=_parse_as_bool(lf['end_cnn'])
    )


def create_MLP_layer(lf, *unused):
    _check_spec([2, 3, 4, 5], lf, 'MLP')
    return MLP(
        input_dim=int(lf['input_dim']),
        output_dim=int(lf['output_dim']),
        activation_type=lf['act'],
        num_hidden_lyr=int(lf['num_hidden_lyr']),
        hidden_channels=_parse_as_int_list(lf["hidden_channels"])
    )


def create_PCA_layer(lf, model, *unused):
    _check_spec([6], lf, 'MLP')
    return PCAModel(desc=[model.num_node_feat] + _parse_as_int_list(lf['desc']),
                    tao1=float(lf['tao1']),
                    tao2=float(lf['tao2']),
                    sinkhorn_iters=int(lf['sinkhorn_iters']),
                    sinkhorn_eps=float(lf['sinkhorn_eps']),
                    aff_max=int(lf['aff_max'])
                    )


def create_OurLossFunction(lf, model, *unused):
    return OurLossFunction(n_features=model.num_node_feat,
                           alpha=float(lf['alpha']),
                           beta=float(lf['beta']),
                           gamma=float(lf['gamma']),
                           tau=float(lf['tau']),
                           y_from=lf['y_from'],
                           z_from=lf['z_from']
                           )


def create_TreeSearch(lf, model, *unused):
    return TreeSearch(n_features=model.num_node_feat,
                      stop_strategy=lf['stop_strategy'],
                      temperature=float(lf['temperature']),
                      loss_opt=lf['loss_opt'])

def create_MatchingMatrixComp(*unused):
    return MatchingMatrixComp()


"""
Register the constructor caller function here.
"""
layer_ctors = {
    'NodeEmbedding': create_NodeEmbedding_layer,
    'NodeEmbeddingCombinator': create_NodeEmbeddingCombinator_layer,
    'NodeEmbeddingInteraction': create_NodeEmbeddingInteraction_layer,
    'Loss': create_Loss_layer,
    'Sequence': create_Sequence_layer,
    'Fancy': create_Fancy_layer,
    'Affinity': create_Affinity_layer,
    'PowerIteration': create_PowerIteration_layer,
    'BiStochastic': create_BiStochastic_layer,
    'OurTransformerPrototype': create_OurTransformerPrototype_layer,
    'GMNEncoder': create_GMNEncoder_layer,
    'GMNPropagator': create_GMNPropagator_layer,
    'GMNAggregator': create_GMNAggregator_layer,
    'GMNLoss': create_GMNLoss_layer,
    'ANPM': create_ANPM_layer,
    'GraphConvolutionCollector': create_GraphConvolutionCollector_layer,
    'CNN': create_CNN_layer,
    'MLP': create_MLP_layer,
    'PCA': create_PCA_layer,
    'OurLossFunction': create_OurLossFunction,
    'TreeSearch': create_TreeSearch,
    'MatchingMatrixComp': create_MatchingMatrixComp
}


def _check_spec(allowed_nums, lf, ln):
    if len(lf) not in allowed_nums:
        raise ValueError('{} layer must have {} specs NOT {} {}'.
                         format(ln, allowed_nums, len(lf), lf))


def _parse_as_bool(b):
    if b == 'True':
        return True
    elif b == 'False':
        return False
    else:
        raise RuntimeError('Unknown bool string {}'.format(b))


def _parse_as_int_list(il):
    rtn = []
    for x in il.split('_'):
        x = int(x)
        rtn.append(x)
    return rtn


'''
from layers import GraphConvolution, GraphConvolutionAttention, Coarsening, Average, \
    Attention, Dot, Dist, NTN, SLM, Dense, Padding, MNE, MNEMatch, CNN, ANPM, ANPMD, ANNH, \
    GraphConvolutionCollector, MNEResize, PadandTruncate, Supersource, JumpingKnowledge
import tensorflow as tf
import numpy as np
from math import exp


def create_layers(model, pattern, num_layers):
    layers = []
    for i in range(1, num_layers + 1):  # 1-indexed
        sp = FLAGS.flag_values_dict()['{}_{}'.format(pattern, i)].split(':')
        name = sp[0]
        layer_info = {}
        if len(sp) > 1:
            assert (len(sp) == 2)
            for spec in sp[1].split(','):
                ssp = spec.split('=')
                layer_info[ssp[0]] = ssp[1]
        if name == 'GraphConvolution':
            layers.append(create_GraphConvolution_layer(layer_info, model, i))
        elif name == 'GraphConvolutionAttention':
            layers.append(create_GraphConvolutionAttention_layer(layer_info, model, i))
        elif name == 'GraphConvolutionCollector':
            layers.append(create_GraphConvolutionCollector_layer(layer_info))
        elif name == 'Coarsening':
            layers.append(create_Coarsening_layer(layer_info))
        elif name == 'Average':
            layers.append(create_Average_layer(layer_info))
        elif name == 'Attention':
            layers.append(create_Attention_layer(layer_info))
        elif name == 'Supersource':
            layers.append(create_Supersource_layer(layer_info))
        elif name == 'JumpingKnowledge':
            layers.append(create_JumpingKnowledge_layer(layer_info))
        elif name == 'Dot':
            layers.append(create_Dot_layer(layer_info))
        elif name == 'Dist':
            layers.append(create_Dist_layer(layer_info))
        elif name == 'SLM':
            layers.append(create_SLM_layer(layer_info))
        elif name == 'NTN':
            layers.append(create_NTN_layer(layer_info))
        elif name == 'ANPM':
            layers.append(create_ANPM_layer(layer_info))
        elif name == 'ANPMD':
            layers.append(create_ANPMD_layer(layer_info))
        elif name == 'ANNH':
            layers.append(create_ANNH_layer(layer_info))
        elif name == 'Dense':
            layers.append(create_Dense_layer(layer_info))
        elif name == 'Padding':
            layers.append(create_Padding_layer(layer_info))
        elif name == 'PadandTruncate':
            layers.append(create_PadandTruncate_layer(layer_info))
        elif name == 'MNE':
            layers.append(create_MNE_layer(layer_info))
        elif name == 'MNEMatch':
            layers.append(create_MNEMatch_layer(layer_info))
        elif name == 'MNEResize':
            layers.append(create_MNEResize_layer(layer_info))
        elif name == 'CNN':
            layers.append(create_CNN_layer(layer_info))
        else:
            raise RuntimeError('Unknown layer {}'.format(name))
    return layers



def create_GraphConvolution_layer(layer_info, model, layer_id):
    if not 5 <= len(layer_info) <= 6:
        raise RuntimeError('GraphConvolution layer must have 3-4 specs')
    input_dim = layer_info.get('input_dim')
    if not input_dim:
        if layer_id != 1:
            raise RuntimeError(
                'The input dim for layer must be specified'.format(layer_id))
        input_dim = model.input_dim
    else:
        input_dim = int(input_dim)
    return GraphConvolution(
        input_dim=input_dim,
        output_dim=int(layer_info['output_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        sparse_inputs=parse_as_bool(layer_info['sparse_inputs']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']),
        featureless=False,
        num_supports=1)


def create_GraphConvolutionAttention_layer(layer_info, model, layer_id):
    if not 5 <= len(layer_info) <= 6:
        raise RuntimeError('GraphConvolution layer must have 3-4 specs')
    input_dim = layer_info.get('input_dim')
    if not input_dim:
        if layer_id != 1:
            raise RuntimeError(
                'The input dim for layer must be specified'.format(layer_id))
        input_dim = model.input_dim
    else:
        input_dim = int(input_dim)
    return GraphConvolutionAttention(
        input_dim=input_dim,
        output_dim=int(layer_info['output_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        sparse_inputs=parse_as_bool(layer_info['sparse_inputs']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']),
        featureless=False,
        num_supports=1)


def create_GraphConvolutionCollector_layer(layer_info):
    if not len(layer_info) == 5:
        raise RuntimeError('GraphConvolutionCollector layer must have 5 spec')
    return GraphConvolutionCollector(gcn_num=int(layer_info['gcn_num']),
                                     fix_size=int(layer_info['fix_size']),
                                     mode=int(layer_info['mode']),
                                     padding_value=int(layer_info['padding_value']),
                                     align_corners=parse_as_bool(layer_info['align_corners']))


def create_Coarsening_layer(layer_info):
    if not len(layer_info) == 1:
        raise RuntimeError('Coarsening layer must have 1 spec')
    return Coarsening(pool_style=layer_info['pool_style'])


def create_Average_layer(layer_info):
    if not len(layer_info) == 0:
        raise RuntimeError('Average layer must have 0 specs')
    return Average()


def create_Attention_layer(layer_info):
    if not len(layer_info) == 5:
        raise RuntimeError('Attention layer must have 5 specs')
    return Attention(input_dim=int(layer_info['input_dim']),
                     att_times=int(layer_info['att_times']),
                     att_num=int(layer_info['att_num']),
                     att_style=layer_info['att_style'],
                     att_weight=parse_as_bool(layer_info['att_weight']))


def create_Supersource_layer(layer_info):
    if not len(layer_info) == 0:
        raise RuntimeError('Supersource layer must have 0 specs')
    return Supersource()


def create_JumpingKnowledge_layer(layer_info):
    if not len(layer_info) == 7:
        raise RuntimeError('JumpingKnowledge layer must have 5 specs')
    return JumpingKnowledge(gcn_num=int(layer_info['gcn_num']),
                            input_dims=parse_as_int_list(layer_info['input_dims']),
                            att_times=int(layer_info['att_times']),
                            att_num=int(layer_info['att_num']),
                            att_style=layer_info['att_style'],
                            att_weight=parse_as_bool(layer_info['att_weight']),
                            combine_method=layer_info['combine_method'])


def create_Dot_layer(layer_info):
    if not len(layer_info) == 2:
        raise RuntimeError('Dot layer must have 2 specs')
    return Dot(output_dim=int(layer_info['output_dim']),
               act=create_activation(layer_info['act']))


def create_Dist_layer(layer_info):
    if not len(layer_info) == 1:
        raise RuntimeError('Dot layer must have 1 specs')
    return Dist(norm=layer_info['norm'])


def create_SLM_layer(layer_info):
    if not len(layer_info) == 5:
        raise RuntimeError('SLM layer must have 5 specs')
    return SLM(
        input_dim=int(layer_info['input_dim']),
        output_dim=int(layer_info['output_dim']),
        act=create_activation(layer_info['act']),
        dropout=parse_as_bool(layer_info['dropout']),
        bias=parse_as_bool(layer_info['bias']))


def create_NTN_layer(layer_info):
    if not len(layer_info) == 6:
        raise RuntimeError('NTN layer must have 6 specs')
    return NTN(
        input_dim=int(layer_info['input_dim']),
        feature_map_dim=int(layer_info['feature_map_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        inneract=create_activation(layer_info['inneract']),
        apply_u=parse_as_bool(layer_info['apply_u']),
        bias=parse_as_bool(layer_info['bias']))


def create_ANPM_layer(layer_info):
    if not len(layer_info) == 14:
        raise RuntimeError('ANPM layer must have 14 specs')
    return ANPM(
        input_dim=int(layer_info['input_dim']),
        att_times=int(layer_info['att_times']),
        att_num=int(layer_info['att_num']),
        att_style=layer_info['att_style'],
        att_weight=parse_as_bool(layer_info['att_weight']),
        feature_map_dim=int(layer_info['feature_map_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        bias=parse_as_bool(layer_info['bias']),
        ntn_inneract=create_activation(layer_info['ntn_inneract']),
        apply_u=parse_as_bool(layer_info['apply_u']),
        padding_value=int(layer_info['padding_value']),
        mne_inneract=create_activation(layer_info['mne_inneract']),
        # num_bins=int(layer_info['num_bins'])
        mne_method=layer_info['mne_method'],
        branch_style=layer_info['branch_style'])


def create_ANPMD_layer(layer_info):
    if not len(layer_info) == 22:
        raise RuntimeError('ANPMD layer must have 22 specs')
    return ANPMD(
        input_dim=int(layer_info['input_dim']),
        att_times=int(layer_info['att_times']),
        att_num=int(layer_info['att_num']),
        att_style=layer_info['att_style'],
        att_weight=parse_as_bool(layer_info['att_weight']),
        feature_map_dim=int(layer_info['feature_map_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        bias=parse_as_bool(layer_info['bias']),
        ntn_inneract=create_activation(layer_info['ntn_inneract']),
        apply_u=parse_as_bool(layer_info['apply_u']),
        padding_value=int(layer_info['padding_value']),
        mne_inneract=create_activation(layer_info['mne_inneract']),
        mne_method=layer_info['mne_method'],
        branch_style=layer_info['branch_style'],
        dense1_dropout=parse_as_bool(layer_info['dense1_dropout']),
        dense1_act=create_activation(layer_info['dense1_act']),
        dense1_bias=parse_as_bool(layer_info['dense1_bias']),
        dense1_output_dim=int(layer_info['dense1_output_dim']),
        dense2_dropout=parse_as_bool(layer_info['dense2_dropout']),
        dense2_act=create_activation(layer_info['dense2_act']),
        dense2_bias=parse_as_bool(layer_info['dense2_bias']),
        dense2_output_dim=int(layer_info['dense2_output_dim']))


def create_ANNH_layer(layer_info):
    if not len(layer_info) == 14:
        raise RuntimeError('ANNH layer must have 14 specs')
    return ANNH(
        input_dim=int(layer_info['input_dim']),
        att_times=int(layer_info['att_times']),
        att_num=int(layer_info['att_num']),
        att_style=layer_info['att_style'],
        att_weight=parse_as_bool(layer_info['att_weight']),
        feature_map_dim=int(layer_info['feature_map_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        bias=parse_as_bool(layer_info['bias']),
        ntn_inneract=create_activation(layer_info['ntn_inneract']),
        apply_u=parse_as_bool(layer_info['apply_u']),
        padding_value=int(layer_info['padding_value']),
        mne_inneract=create_activation(layer_info['mne_inneract']),
        # num_bins=int(layer_info['num_bins'])
        mne_method=layer_info['mne_method'],
        branch_style=layer_info['branch_style'])


def create_Dense_layer(layer_info):
    if not len(layer_info) == 5:
        raise RuntimeError('Dense layer must have 5 specs')
    return Dense(
        input_dim=int(layer_info['input_dim']),
        output_dim=int(layer_info['output_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']))


def create_Padding_layer(layer_info):
    if not len(layer_info) == 1:
        raise RuntimeError('Padding layer must have 1 specs')
    return Padding(
        padding_value=int(layer_info['padding_value']))


def create_PadandTruncate_layer(layer_info):
    if not len(layer_info) == 1:
        raise RuntimeError('PadandTruncate layer must have 1 specs')
    return PadandTruncate(
        padding_value=int(layer_info['padding_value']))


def create_MNE_layer(layer_info):
    if not len(layer_info) == 3:
        raise RuntimeError('MNE layer must have 3 specs')
    return MNE(
        input_dim=int(layer_info['input_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        inneract=create_activation(layer_info['inneract']))


def create_MNEMatch_layer(layer_info):
    if not len(layer_info) == 2:
        raise RuntimeError('MNEMatch layer must have 2 specs')
    return MNEMatch(
        input_dim=int(layer_info['input_dim']),
        inneract=create_activation(layer_info['inneract']))


def create_MNEResize_layer(layer_info):
    if not len(layer_info) == 6:
        raise RuntimeError('MNEResize layer must have 6 specs')
    return MNEResize(
        dropout=parse_as_bool(layer_info['dropout']),
        inneract=create_activation(layer_info['inneract']),
        fix_size=int(layer_info['fix_size']),
        mode=int(layer_info['mode']),
        padding_value=int(layer_info['padding_value']),
        align_corners=parse_as_bool(layer_info['align_corners']))


def create_CNN_layer(layer_info):
    if not 11 <= len(layer_info) <= 13:
        raise RuntimeError('CNN layer must have 11-13 specs')
    gcn_num = layer_info.get('gcn_num')
    mode = layer_info.get('mode')
    if not gcn_num:
        if layer_info['mode'] != 'merge':
            raise RuntimeError('The gcn_num for layer must be specified')
        gcn_num = None
    else:
        gcn_num = int(gcn_num)
    mode = 'merge' if not mode else mode

    return CNN(
        start_cnn=parse_as_bool(layer_info['start_cnn']),
        end_cnn=parse_as_bool(layer_info['end_cnn']),
        window_size=int(layer_info['window_size']),
        kernel_stride=int(layer_info['kernel_stride']),
        in_channel=int(layer_info['in_channel']),
        out_channel=int(layer_info['out_channel']),
        padding=layer_info['padding'],
        pool_size=int(layer_info['pool_size']),
        dropout=parse_as_bool(layer_info['dropout']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']),
        mode=mode,
        gcn_num=gcn_num)


def create_activation(act, ds_kernel=None, use_tf=True):
    if act == 'relu':
        return tf.nn.relu if use_tf else relu_np
    elif act == 'identity':
        return tf.identity if use_tf else identity_np
    elif act == 'sigmoid':
        return tf.sigmoid if use_tf else sigmoid_np
    elif act == 'tanh':
        return tf.tanh if use_tf else np.tanh
    elif act == 'ds_kernel':
        return ds_kernel.dist_to_sim_tf if use_tf else \
            ds_kernel.dist_to_sim_np
    else:
        raise RuntimeError('Unknown activation function {}'.format(act))


def relu_np(x):
    return np.maximum(x, 0)


def identity_np(x):
    return x


def sigmoid_np(x):
    try:
        ans = exp(-x)
    except OverflowError:  # TODO: fix
        ans = float('inf')
    return 1 / (1 + ans)


def parse_as_bool(b):
    if b == 'True':
        return True
    elif b == 'False':
        return False
    else:
        raise RuntimeError('Unknown bool string {}'.format(b))


def parse_as_int_list(il):
    rtn = []
    for x in il.split('_'):
        x = int(x)
        rtn.append(x)
    return rtn
'''
