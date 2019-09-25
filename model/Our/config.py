from solve_parent_dir import solve_parent_dir
from dataset_config import get_dataset_conf
from dist_sim import get_ds_metric_config
from utils import format_str_list, C, get_user, get_host
import argparse
import torch

solve_parent_dir()
parser = argparse.ArgumentParser()

"""
Data.
"""

""" 
dataset: 
    (for MCS)
    aids700nef linux imdbmulti redditmulti10k
"""
dataset = 'aids700nef'
parser.add_argument('--dataset', default=dataset)

filter_large_size = None
parser.add_argument('--filter_large_size', type=int, default=filter_large_size)  # None or >= 1

select_node_pair = None
parser.add_argument('--select_node_pair', type=str, default=select_node_pair)  # None or gid1_gid2

c = C()
parser.add_argument('--node_fe_{}'.format(c.c()), default='one_hot')

# parser.add_argument('--node_fe_{}'.format(c.c()),
#                     default='local_degree_profile')

natts, eatts, tvt_options, align_metric_options, *_ = \
    get_dataset_conf(dataset)

""" Must use exactly one alignment metric across the entire run. """
align_metric = align_metric_options[0]
if len(align_metric_options) == 2:
    """ Choose which metric to use. """
    align_metric = 'ged'
    # align_metric = 'mcs'
parser.add_argument('--align_metric', default=align_metric)

dos_true, _ = get_ds_metric_config(align_metric)
parser.add_argument('--dos_true', default=dos_true)

# Assume the model predicts None. May be updated below.
dos_pred = None

parser.add_argument('--node_feats', default=format_str_list(natts))

parser.add_argument('--edge_feats', default=format_str_list(eatts))
"""
Evaluation.
"""

parser.add_argument('--tvt_options', default=format_str_list(tvt_options))

""" holdout, (TODO) <k>-fold. """
tvt_strategy = 'holdout'
parser.add_argument('--tvt_strategy', default=tvt_strategy)

if tvt_strategy == 'holdout':
    if tvt_options == ['all']:
        parser.add_argument('--train_test_ratio', type=float, default=0.8)
    elif tvt_options == ['train', 'test']:
        pass
    else:
        raise NotImplementedError()
else:
    raise NotImplementedError()

parser.add_argument('--debug', type=bool, default='debug' in dataset)

# Assume normalization is needed for true dist/sim scores.
# parser.add_argument('--ds_norm', type=bool, default=True)

# parser.add_argument('--ds_kernel', default='exp_0.7')

"""
Model.
"""

"""
# Legacy Models
# model = 'basic'
# model = 'gmn_icml_mlp_simdist'
# model = 'gmn_icml_mlp_mcs'
# model = 'sequence'
# model = 'sequence_transformer'
# model = 'prototype_transformer'
# model = 'model_4'
"""
# model = 'GMN_cvpr-BCE'
# model = 'PCA-BCE'
# model = 'GAT-BCE'
# model = 'GAT-OurLoss'
# model = 'GMN_icml_mlp-BCE'
# model = 'GMN_icml_mlp-OurLoss' # (same as gmn_icml_mlp_mcs)
model = 'GMN_icml_mlp-OurLoss-tree_search'
# model = 'GMN_icml_mlp-tree_search-OurLoss'  # tree_search is part of end-to-end
parser.add_argument('--model', default=model)

loss_opt = None

n_outputs = 3 # TODO: tune this
parser.add_argument('--n_outputs', type=int, default=n_outputs)

hard_mask = True
parser.add_argument('--hard_mask', type=bool, default=hard_mask)

model_name = 'fancy'
parser.add_argument('--model_name', default=model_name)

c = C()

D = 32 

if dataset == 'aids700nef':
    alpha = 1  # 0.01
    beta = 0  # 0.01
    gamma = 0  # 0.2
    tau = 0  # 1
    theta = 0.5  # 0.5
elif dataset == 'linux':
    alpha = 1  # 5
    beta = 0  # 25
    gamma = 0  # 100
    tau = 0  # 1
    theta = 0.5  # 0.5
elif dataset == 'imdbmulti':
    alpha = 1  # 1
    beta = 0  # 2
    gamma = 0  # 0.1
    tau = 0  # 1
    theta = 0.7  # 0.7
elif dataset == 'redditmulti10k':
    alpha = 1  # 10
    beta = 0  # 10
    gamma = 0  # 50
    tau = 0  # 1
    theta = 0.5  # 0.5
else:
    alpha = 1  # 1
    beta = 0  # 0
    gamma = 0  # 0
    tau = 0  # 1
    theta = 0.5  # 0.5
    # assert False

parser.add_argument('--theta', type=float, default=theta)

########################################
# Node Embedding
########################################
combination_layers = None
if 'GAT' in model:
    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gat,output_dim={},act=relu,bn=False'.format(D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gat,input_dim={},output_dim={},act=relu,bn=False'. \
        format(D, D)  # D // 2)
    # D //= 2
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gat,input_dim={},output_dim={},act=identity,bn=False'. \
        format(D, D)  # D // 2)
    # D //= 2
    parser.add_argument(n, default=s)

    combination_layers = '0_1_2_3'
elif 'GMN_icml_mlp' in model:
    n = '--layer_{}'.format(c.c())
    s = 'GMNEncoder:output_dim={},act=relu'.format(D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine'.format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine'.format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine'.format(D, D)
    parser.add_argument(n, default=s)

    combination_layers = '0_2_3_4'
elif 'GMN_icml_gru' in model:
    n = '--layer_{}'.format(c.c())
    s = 'GMNEncoder:output_dim={},act=relu,f_node=GRU'.format(D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,f_node=GRU'.format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,f_node=GRU'.format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'GMNPropagator:input_dim={},' \
        'output_dim={},distance_metric=cosine,f_node=GRU'.format(D, D)
    parser.add_argument(n, default=s)
    combination_layers = '0_2_3_4'
elif 'GMN_cvpr' in model:
    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gat,output_dim={},act=relu,bn=False'.format(D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gat,input_dim={},output_dim={},act=relu,bn=False'. \
        format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbedding:type=gat,input_dim={},output_dim={},act=identity,bn=False'. \
        format(D, D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'Affinitoy:F=1,U=2'.format(D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'PowerIteration:k=10'.format(D)
    parser.add_argument(n, default=s)

    n = '--layer_{}'.format(c.c())
    s = 'BiStochastic:k=10'.format(D)
    parser.add_argument(n, default=s)
    # TODO: do not include node combinator for GMN-CVPR
    model = '{}-raw'.format(model)
elif 'PCA' in model:
    n = '--layer_{}'.format(c.c())
    s = 'PCA:desc=64_64_32,tao1=0.001,tao2=0.001,' \
        'sinkhorn_iters=5,sinkhorn_eps=1e-6,aff_max=85'.format(D)
    parser.add_argument(n, default=s)
    # TODO: do not include node combinator for PCA
    model = '{}-raw'.format(model)

else:
    raise NotImplementedError

########################################
# Node Combinator
########################################
if 'raw' in model:
    pass
else:
    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbeddingCombinator:from_layers={},style=mlp'.format(combination_layers)
    parser.add_argument(n, default=s)

########################################
# Node Interaction
########################################
if 'PCA' not in model and 'GMN_cvpr' not in model:
    n = '--layer_{}'.format(c.c())
    s = 'NodeEmbeddingInteraction:type=dot,input_dim={}'.format(D)
    parser.add_argument(n, default=s)

########################################
# Matching Matrix Computation (Probability Interpretation for Node Interaction)
########################################
if 'PCA' not in model and 'GMN_cvpr' not in model:
    n = '--layer_{}'.format(c.c())
    s = 'MatchingMatrixComp'
    parser.add_argument(n, default=s)

########################################
# Node Loss Function
########################################
if 'BCE' in model:
    n = '--layer_{}'.format(c.c())
    s = 'Loss:type=BCEWithLogitsLoss'
    parser.add_argument(n, default=s)


elif 'OurLoss' in model:
    if 'OurLoss-tree_search' in model:  # from yz generated by MatchingMatrixComp
        y_from = 'y_pred'
        z_from = 'z_pred'
        n = '--layer_{}'.format(c.c())
    elif 'tree_search-OurLoss' in model:  # from yz generated by TreeSearch
        # y_from = 'tree_pred_hard'
        # z_from = 'hidden_state_v_hard'
        y_from = 'tree_pred_soft'
        z_from = 'hidden_state_v_soft'
        n = '--layer_{}'.format(c.c() + 1)
    else:
        assert False
    s = 'OurLossFunction:alpha={},beta={},gamma={},tau={},y_from={},z_from={}'.format(
        alpha, beta, gamma, tau, y_from, z_from)
    parser.add_argument(n, default=s)
else:
    raise NotImplementedError

########################################
# Tree Search
########################################
if 'tree_search' in model:
    if 'OurLoss-tree_search' in model:
        n = '--layer_{}'.format(c.c())
    elif 'tree_search-OurLoss' in model:
        n = '--layer_{}'.format(c.c() - 1)
    else:
        assert False
    # stop_strategy = 'threshold_0.001'
    stop_strategy = 'sin_eps=0.0001_D=16'
    # nonlearnable = False
    temperature = 0.1
    s = 'TreeSearch:stop_strategy={},temperature={},loss_opt={}'.format(
        stop_strategy, temperature, loss_opt)
    parser.add_argument(n, default=s)

parser.add_argument('--layer_num', type=int, default=c.t())

# Finally we set dos_pred.
parser.add_argument('--dos_pred', default=dos_pred)

"""
Optimization.
"""

lr = 1e-3
parser.add_argument('--lr', type=float, default=lr)

gpu = -1
device = str('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1
             else 'cpu')
parser.add_argument('--device', default=device)

# num_epochs = 1
# parser.add_argument('--num_epochs', type=int, default=num_epochs)

'''
lmbda = 1.0
parser.add_argument('--lmbda', type=float, default=lmbda)
'''

num_iters = 3000  # TODO: tune this
parser.add_argument('--num_iters', type=int, default=num_iters)

validation = False  # TODO: tune this
parser.add_argument('--validation', type=bool, default=validation)

throw_away = 0  # TODO: tune this
parser.add_argument('--throw_away', type=float, default=throw_away)

print_every_iters = 10
parser.add_argument('--print_every_iters', type=int, default=print_every_iters)

only_iters_for_debug = None  # only train and test this number of pairs
parser.add_argument('--only_iters_for_debug', type=int, default=only_iters_for_debug)

save_model = False  # TODO: tune this
parser.add_argument('--save_model', type=bool, default=save_model)

batch_size = 64 
parser.add_argument('--batch_size', type=int, default=batch_size)

parser.add_argument('--node_ordering', default='bfs')
parser.add_argument('--no_probability', default=False)
parser.add_argument('--positional_encoding', default=False)  # TODO: dataset.py cannot see this

"""
Other info.
"""
parser.add_argument('--user', default=get_user())

parser.add_argument('--hostname', default=get_host())

FLAGS = parser.parse_args()
