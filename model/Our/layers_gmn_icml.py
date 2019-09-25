from config import FLAGS
import torch.nn as nn
from layers import create_act
from torch_scatter import scatter_add
import torch


class MLP(nn.Module):
    '''mlp can specify number of hidden layers and hidden layer channels'''

    def __init__(self, input_dim, output_dim, activation_type='relu', num_hidden_lyr=2, hidden_channels=None):
        super().__init__()
        self.out_dim = output_dim
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError("number of hidden layers should be the same as the lengh of hidden_channels")
        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.activation = create_act(activation_type)
        self.layers = nn.ModuleList(list(map(self.weight_init,[nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                                     for i in range(len(self.layer_channels) - 1)])))
    def weight_init(self, m):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        return m

    def forward(self, x, batch_data, model):
        layer_inputs = [x]
        for layer in self.layers:
            input = layer_inputs[-1]
            layer_inputs.append(self.activation(layer(input)))
        model.store_layer_output(self, layer_inputs[-1])
        return layer_inputs[-1]


class GMNPropagator(nn.Module):
    def __init__(self, input_dim, output_dim, distance_metric='cosine', f_node='MLP'):
        super().__init__()
        self.out_dim = output_dim
        if distance_metric == 'cosine':
            self.distance_metric = nn.CosineSimilarity()
        elif distance_metric == 'euclidean':
            self.distance_metric = nn.PairwiseDistance()
        self.softmax = nn.Softmax(dim=1)
        self.f_messasge = MLP(2 * input_dim, 2 * input_dim, num_hidden_lyr=1, hidden_channels=[2 * input_dim]) #2*input_dim because in_dim = dim(g1) + dim(g2)
        self.f_node_name = f_node
        if f_node=='MLP':
            self.f_node = MLP(4 * input_dim, output_dim, num_hidden_lyr=1) #2*input_dim for m_sum, 1 * input_dim for u_sum and 1*input_dim for x
        elif f_node=='GRU':
            self.f_node = nn.GRUCell(3 *input_dim, input_dim)# 2*input_dim for m_sum, 1 * input_dim for u_sum
        else:
            raise ValueError("{} for f_node has not been implemented".format(f_node))


    def forward(self, ins, batch_data, model):
        x = ins  # x has shape N(gs) by D
        edge_index = batch_data.merge_data['merge'].edge_index #edges of each graph
        row, col = edge_index
        m = torch.cat((x[row], x[col]), dim=1)  # E by (2 * D)
        m = self.f_messasge(m, batch_data, model)
        m_sum = scatter_add(m, row, dim=0, dim_size=x.size(0))  # N(gs) by (2 * D)
        u_sum = self.f_match(x, batch_data)  # u_sum has shape N(gs) by D

        if self.f_node_name == 'MLP':
            in_f_node = torch.cat((x, m_sum, u_sum), dim=1)
            out = self.f_node(in_f_node, batch_data, model) 
        elif self.f_node_name == 'GRU':
            in_f_node = torch.cat((m_sum, u_sum), dim=1) # N by 3*D
            out = self.f_node(in_f_node, x)

        model.store_layer_output(self, out)
        return out

    def f_match(self, x, batch_data):
        '''from the paper https://openreview.net/pdf?id=S1xiOjC9F7'''
        ind_list = batch_data.merge_data['ind_list']
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

        sum_a1_h = torch.sum(g2x * a1[:, :, None], dim=1)  # N1 by D tensor where each row is sum_j(a_j * h_j)
        return g1x - sum_a1_h


class GMNAggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.out_dim = output_dim
        self.sigmoid = nn.Sigmoid()
        self.weight_func = MLP(input_dim, output_dim, num_hidden_lyr=1, hidden_channels=[output_dim])
        self.gate_func = MLP(input_dim, output_dim, num_hidden_lyr=1, hidden_channels=[output_dim])
        self.mlp_graph = MLP(output_dim, output_dim, num_hidden_lyr=1, hidden_channels=[output_dim])

    def forward(self, x, batch_data, model):
        weighted_x = self.weight_func(x, batch_data, model)  # shape N by input_dim
        gated_x = self.sigmoid(self.gate_func(x, batch_data, model))  # shape N by input_dim
        hammard_prod = gated_x * weighted_x
        merge_data = batch_data.merge_data['merge']
        batch = merge_data.batch
        num_graphs = merge_data.num_graphs

        graph_embeddings = scatter_add(hammard_prod, batch, dim=0, dim_size=num_graphs)  # shape G by output_dim
        return self.mlp_graph(graph_embeddings, batch_data, model)


class GMNLoss(nn.Module):
    def __init__(self, ds_metric='cosine'):
        super().__init__()
        if ds_metric == 'cosine':
            self.ds_metric = nn.CosineSimilarity()
            if FLAGS.dos_pred != 'sim':
                raise ValueError('cosine must use dos_pred == sim')
        elif ds_metric == 'euclidean':
            self.ds_metric = nn.PairwiseDistance()
            if FLAGS.dos_pred != 'euclidean':
                raise ValueError('euclidean must use dos_pred == dist')
        elif ds_metric == 'scalar':
            self.ds_metric = None
        self.loss = nn.MSELoss()

    def forward(self, x, batch_data, model):

        true_score = [(pair.get_ds_true(
            FLAGS.ds_norm, FLAGS.dos_true, FLAGS.dos_pred, FLAGS.ds_kernel)) for pair in
            batch_data.pair_list]
        if self.ds_metric:
            g1s = x[0::2, ]  # num_graphs/2 by D
            g2s = x[1::2, ]
            pred_ds = self.ds_metric(g1s, g2s)
        else:
            pred_ds = x.squeeze()
            sum_preds = sum(pred_ds)
            if sum_preds / len(x) < 0.05:
                print("Weights poorly initialized please try running again")
                exit(1)

        assert len(true_score) == pred_ds.size(0)

        for i, pair in enumerate(batch_data.pair_list):
            pair.assign_ds_pred(pred_ds[i])

        return self.loss(pred_ds, torch.tensor(true_score, device=FLAGS.device))

# TODO implement loss function
