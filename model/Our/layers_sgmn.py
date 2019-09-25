from config import FLAGS
import torch.nn as nn
from layers import create_act
from torch_scatter import scatter_add
import torch


class MLP(nn.Module):
    '''mlp can specify number of hidden layers and hidden layer channels'''

    def __init__(self, input_dim, output_dim, activation_type='relu', num_hidden_lyr=2,
                 hidden_channels=None):
        super().__init__()
        self.out_dim = output_dim
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError(
                "number of hidden layers should be the same as the lengh of hidden_channels")
        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.activation = create_act(activation_type)
        self.layers = nn.ModuleList(list(
            map(self.weight_init, [nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                                   for i in range(len(self.layer_channels) - 1)])))

    def weight_init(self, m):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        return m

    def forward(self, x):
        layer_inputs = [x]
        for layer in self.layers:
            input = layer_inputs[-1]
            layer_inputs.append(self.activation(layer(input)))
        return layer_inputs[-1]


class SGMNPropagator():
    def __init__(self, input_dim, output_dim, distance_metric='cosine', f_node='MLP'):
        super(SGMNPropagator).__init__()
        self.out_dim = output_dim

        self.red_embedding = torch.nn.Parameter(torch.randn(input_dim)).to(FLAGS.device)
        self.orange_embedding = torch.nn.Parameter(torch.randn(input_dim)).to(FLAGS.device)
        self.gray_embedding = torch.nn.Parameter(torch.randn(input_dim)).to(FLAGS.device)

        if distance_metric == 'cosine':
            self.distance_metric = nn.CosineSimilarity()
        elif distance_metric == 'euclidean':
            self.distance_metric = nn.PairwiseDistance()
        self.softmax = nn.Softmax(dim=1)
        self.f_messasge = MLP(2 * input_dim, 2 * input_dim, num_hidden_lyr=1, hidden_channels=[
            2 * input_dim])  # 2*input_dim because in_dim = dim(g1) + dim(g2)
        self.f_node_name = f_node
        if f_node == 'MLP':
            self.f_node = MLP(4 * input_dim, output_dim,
                              num_hidden_lyr=1)  # 2*input_dim for m_sum, 1 * input_dim for u_sum and 1*input_dim for x
        elif f_node == 'GRU':
            self.f_node = nn.GRUCell(3 * input_dim,
                                     input_dim)  # 2*input_dim for m_sum, 1 * input_dim for u_sum
        else:
            raise ValueError("{} for f_node has not been implemented".format(f_node))

    def __call__(self, sgmn_ins_1, sgmn_ins_2):
        xs = []
        m_sums = []
        ind_list = []
        cum_sum = 0
        for sgmn_ins in [sgmn_ins_1, sgmn_ins_2]:
            x, edge_index, red_indicator, orange_indicator, gray_indicator = sgmn_ins
            N, D = x.shape
            positional_encoding = torch.ger(red_indicator, self.red_embedding) + \
                                  torch.ger(orange_indicator, self.orange_embedding) + \
                                  torch.ger(gray_indicator, self.gray_embedding)
            x = x + positional_encoding  # x has shape N(gs) by D
            # edge_index = batch_data.merge_data['merge'].edge_index  # edges of each graph
            row, col = edge_index
            m = torch.cat((x[row], x[col]), dim=1)  # E by (2 * D)
            m = self.f_messasge(m)
            m_sum = scatter_add(m, row, dim=0, dim_size=x.size(0))  # N(gs) by (2 * D)
            ind_list.append((cum_sum, cum_sum+N))
            xs.append(x)
            m_sums.append(m_sum)
            cum_sum += N
        assert len(xs) == 2 and len(m_sums) == 2
        m_sum = torch.cat((m_sums[0], m_sums[1]),dim=0)
        x = torch.cat((xs[0], xs[1]),dim=0)
        u_sum = self.f_match(x, ind_list)  # u_sum has shape N(gs) by D

        if self.f_node_name == 'MLP':
            in_f_node = torch.cat((x, m_sum, u_sum), dim=1)
            out = self.f_node(in_f_node)
        elif self.f_node_name == 'GRU':
            in_f_node = torch.cat((m_sum, u_sum), dim=1)  # N by 3*D
            out = self.f_node(in_f_node, x)

        assert len(ind_list) == 2
        split = ind_list[0][1]
        out1 = out[:split]
        out2 = out[split:]
        return out1, out2

    def f_match(self, x, ind_list):
        '''from the paper https://openreview.net/pdf?id=S1xiOjC9F7'''
        #ind_list = batch_data.merge_data['ind_list']
        u_all_l = []

        for i in range(0, len(ind_list), 2):
            g1_ind = i
            g2_ind = i + 1
            g1x = x[ind_list[g1_ind][0]: ind_list[g1_ind][1]]
            g2x = x[ind_list[g2_ind][0]: ind_list[g2_ind][1]]

            u1 = self._f_match_helper(g1x, g2x)  # N(g1) by D tensor
            u2 = self._f_match_helper(g2x, g1x)  # N(g2) by D tensor

            u_all_l.append(u1)
            u_all_l.append(u2)

        return torch.cat(u_all_l, dim=0).view(x.size(0), -1)

    def _f_match_helper(self, g1x, g2x):
        g1_norm = torch.nn.functional.normalize(g1x, p=2, dim=1)
        g2_norm = torch.nn.functional.normalize(g2x, p=2, dim=1)
        g1_sim = torch.matmul(g1_norm, torch.t(g2_norm))

        # N_1 by N_2 tensor where a1[x][y] is the softmaxed a_ij of the yth node of g2 to the xth node of g1
        a1 = self.softmax(g1_sim)

        sum_a1_h = torch.sum(g2x * a1[:, :, None],
                             dim=1)  # N1 by D tensor where each row is sum_j(a_j * h_j)
        return g1x - sum_a1_h
