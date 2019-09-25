from layers_sgmn import SGMNPropagator
from batch import create_edge_index
from utils_our import debug_tensor
from dataset_config import get_dataset_conf
from load_data import load_dataset
from utils import get_model_path, load
from os.path import join
from config import FLAGS
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, RGCNConv
from torch.autograd import Variable


class State(object):
    def __init__(self, y_pred_mat, graph_pair):
        self.y_pred_mat = y_pred_mat  # TODO: in future check if it is updated since it is a pointer (eval)
        # self.match_mat = torch.zeros(y_pred_mat.shape)  # 0/1 matching matrix to return
        self.graph_pair = graph_pair
        g1, g2 = graph_pair.get_g1_g2()
        self.adj_mat1 = self._get_adjmat(g1)
        self.adj_mat2 = self._get_adjmat(g2)
        M, N = y_pred_mat.shape
        ### Hard version
        self.v1 = torch.zeros(M, device=FLAGS.device)  # indicator vector of node selection
        self.v2 = torch.zeros(N, device=FLAGS.device)
        self.tree_pred = torch.zeros((M, N), device=FLAGS.device)
        '''
        #############################
        # FOR PROTOTYPING!!!
        ### Soft version
        self.v1_soft = torch.zeros(M, device=FLAGS.device)
        self.v2_soft = torch.zeros(N, device=FLAGS.device)
        self.tree_pred_soft = torch.zeros((M, N), device=FLAGS.device)
        self.soft_score = 0.0  # indicate how likely the search result is optimal
        #############################
        '''
        ### y_pred mask representing the search frontier.
        self.frontier_mask = torch.ones((M, N), device=FLAGS.device)

    def _get_adjmat(self, g):
        coo = nx.adjacency_matrix(g.nxgraph).tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense().to(
            device=FLAGS.device)

    def get_match_mat(self):  # infer 0/1 matching matrix from v1 and v2
        # M, N = self.y_pred_mat.shape
        # v1 = self.v1.repeat(N, 1)
        # v2 = self.v2.repeat(M, 1)
        # y_pred_mat = torch.t(v1) * v2
        # return y_pred_mat  # TODO: 0/1 matching matrix to return
        return self.tree_pred

    def commit_pair_pick(self, update_material):
        # v1_pair, v2_pair, tree_pred_pair, v1_pair_soft, v2_pair_soft, \
        # tree_pred_pair_soft, y_pred_mat_masked = update_material
        v1_pair, v2_pair, tree_pred_pair = update_material
        self.v1 += v1_pair
        self.v2 += v2_pair
        self.tree_pred += tree_pred_pair
        '''
        #############################
        # FOR PROTOTYPING!!!
        self.v1_soft += v1_pair_soft
        self.v2_soft += v2_pair_soft
        self.tree_pred_soft += tree_pred_pair_soft
        #############################
        '''

    def get_x_edge_index(self):
        if not (hasattr(self, 'x1') and hasattr(self, 'x2') and
                hasattr(self, 'edge_index1') and hasattr(self, 'edge_index2')):
            pair = self.graph_pair
            g1, g2 = pair.g1.get_nxgraph(), pair.g2.get_nxgraph()
            cvt_init_x = lambda x: torch.tensor(x, dtype=torch.float).to(FLAGS.device)
            self.x1, self.x2 = cvt_init_x(g1.init_x), cvt_init_x(g2.init_x)
            self.edge_index1, self.edge_index2 = create_edge_index(g1), create_edge_index(g2)
        return self.x1, self.x2, self.edge_index1, self.edge_index2

class TreeSearch(nn.Module):
    def __init__(self, n_features, stop_strategy, temperature, loss_opt):
        super(TreeSearch, self).__init__()
        # self.beam_size = beam_size
        # self.num_soln = num_soln
        self.stop_strategy = stop_strategy

        ### store loss
        # if None, we just pass on the loss from previous layer
        # else compute loss at each iteration and sum
        assert loss_opt in ['None', 'tree_hard', 'tree_soft', 'y_mask']
        self.loss_opt = loss_opt
        self.loss = 0

        if 'sin' in self.stop_strategy:
            nn1 = lambda x: x  # nn.Sequential(nn.Linear(n_features, n_features), nn.ReLU())
            nn2 = lambda x: x  # nn.Sequential(nn.Linear(n_features, n_features), nn.ReLU())
            nn3 = lambda x: x  # nn.Sequential(nn.Linear(n_features, n_features), nn.ReLU())
            if self.loss_opt != 'None':  # use learnable stopping condition if loss_opt != None
                pass
                '''
                #############################
                # FOR PROTOTYPING!!!
                D = int(self.stop_strategy.split('_')[2].replace('D=', ''))
                nn1 = nn.Sequential(nn.Linear(n_features, D), nn.ReLU())
                nn2 = nn.Sequential(nn.Linear(D, D), nn.ReLU())
                nn3 = nn.Sequential(nn.Linear(D, D), nn.ReLU())
                #############################
                '''
            self.eps = float(self.stop_strategy.split('_')[1].replace('eps=', ''))
        else:
            nn1 = lambda x: x  # nn.Sequential(nn.Linear(n_features, n_features), nn.ReLU())
            nn2 = lambda x: x  # nn.Sequential(nn.Linear(n_features, n_features), nn.ReLU())
            nn3 = lambda x: x  # nn.Sequential(nn.Linear(n_features, n_features), nn.ReLU())
            self.eps = None
            D = None

        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.conv3 = GINConv(nn3)

        self.temperature = temperature
        assert self.temperature > 0
        # print('@@@@@', self.temperature)

        self.sgmn = SGMNPropagator(29, 29)

    def forward(self, ins, batch_data, model):
        pair_list = batch_data.pair_list
        for i, pred_pair in enumerate(pair_list):
            tree_pred_mat_list, state_list = self.tree_search_find_mcs(pred_pair)
            pred_pair.assign_tree_pred_list(  # used by evaluation
                tree_pred_mat_list, format='torch_{}'.format(FLAGS.device))
            # pred_pair.state_list = state_list  # IMPORTANT: used by OurLossFunction TODO: check memory usage
        if self.loss_opt == 'None':
            # This will make both OurLoss -> TreeSearch and TreeSearch -> OurLoss work,
            # because:
            # (1) OurLoss -> TreeSearch: TreeSearch only serves as a subgraph
            #     extraction method which is NOT part of the end-to-end process.
            #     So it simply returns the loss generated by prev layer
            #     by returning ins
            # (2) TreeSearch -> OurLoss: TreeSearch lets OurLoss handle the loss
            #     by assigning state_list to pair objects.
            #     OurLoss does not use
            #     ins, but rather tree_pred_mat or y_pred_mat
            #     In other words, ins is only useful for simple sequential models,
            #     e.g. the GCN-GCN-GCN stack.
            #     As models get more flexible and complex, we rely on the
            #     pair objects to pass information between layers.
            self.loss = ins
        else:
            assert NotImplementedError
        return self.loss

    def tree_search_find_mcs(self, pred_pair):
        # assert type(self.beam_size) is int and (self.beam_size >= 2 or self.beam_size == -1)
        # assert type(self.num_soln) is int and self.num_soln >= 1
        tree_pred_mat_list, state_list = [], []
        y_pred_mats = pred_pair.get_y_pred_list_mat_view(format='torch_{}'.format(FLAGS.device))
        for y_pred_mat in y_pred_mats:
            tree_pred_mat, state = self._tree_search_y_pred_mat(pred_pair, y_pred_mat)
            tree_pred_mat_list.append(tree_pred_mat)
            # state_list.append(state) TODO: if saving state list uncomment this line!!
        return tree_pred_mat_list, state_list

    def _tree_search_y_pred_mat(self, pair, y_pred_mat):
        state = State(y_pred_mat, pair)  # TODO: multiple solns
        while self._try_selecting_node_pair(state):
            # Only form a mask when not done, i.e. selection is a success.
            self._expand_search_frontier(state, pair)
        return state.get_match_mat(), state  # TODO

    def _try_selecting_node_pair(self, state):  # select if possible and return success or not
        y_pred_mat = state.y_pred_mat
        mask = state.frontier_mask
        update_material = self._select_pair_without_commit(state)
        # TODO: check max length reached
        if 'threshold' in self.stop_strategy:
            thresh = float(self.stop_strategy.split('_')[1])
            assert thresh >= 0 and thresh <= 1
            if torch.max(mask * y_pred_mat) < thresh:
                return False
        elif 'sin' in self.stop_strategy:
            # Assume all #s of y_pred_mat are in [0, 1].
            if torch.max(mask * y_pred_mat) < 1e-6:  # if all #s are 0, stop
                return False
            x1, x2, edge_index1, edge_index2 = state.get_x_edge_index()
            # Try selecting a new pair but do NOT commit yet.
            # v1_pair, v2_pair, _, _, _, _, _ = update_material
            v1_pair, v2_pair, _ = update_material
            # Do a subgraph isomorphism check.
            q1 = self._get_subgraph_embedding(x1, state.v1 + v1_pair, edge_index1)
            q2 = self._get_subgraph_embedding(x2, state.v2 + v2_pair, edge_index2)
            '''
            # debugging code
            numpy_v1_pair = v1_pair.data.numpy()
            numpy_v2_pair = v2_pair.data.numpy()
            numpy_v1 = state.v1.data.numpy()
            numpy_v2 = state.v2.data.numpy()
            numpy_q1 = q1.data.numpy()
            numpy_q2 = q2.data.numpy()
            '''
            delta = torch.norm(q1 - q2, p=2)
            if delta > self.eps:
                '''
                # debugging code
                lll = numpy_q1 - numpy_q2
                import matplotlib.pyplot as plt
                plt.subplot(2,2,1)
                nx.draw(g1, with_labels = True)
                plt.subplot(2,2,2)
                nx.draw(g2, with_labels = True)
                sub_g1 = g1.copy()
                sub_g2 = g2.copy()
                for i,v1_elt in enumerate(numpy_v1_pair + numpy_v1):
                    if v1_elt == 0:
                        sub_g1.remove_node(i)
                for i,v2_elt in enumerate(numpy_v2_pair + numpy_v2):
                    if v2_elt == 0:
                        sub_g2.remove_node(i)
                plt.subplot(2,2,3)
                nx.draw(sub_g1, with_labels = True)
                plt.subplot(2,2,4)
                nx.draw(sub_g2, with_labels = True)
                plt.show()
                '''
                return False
        else:
            raise NotImplementedError()
        # Passed the check.
        # Commit.
        state.commit_pair_pick(update_material)
        if self.loss_opt != 'None':
            self._update_loss(update_material)
        return True

    def _select_pair_without_commit(
            self, state):  # just pick the pair without modifying state (don't commit yet)
        y_pred_mat_masked = state.y_pred_mat * state.frontier_mask
        M, N = state.y_pred_mat.shape
        v1_pair = torch.zeros(M, device=FLAGS.device)
        v2_pair = torch.zeros(N, device=FLAGS.device)
        tree_pred_pair = torch.zeros((M, N), device=FLAGS.device)
        # non-differentiable version
        indx = torch.argmax(y_pred_mat_masked)
        i, j = int(indx / N), int(indx % N)
        v1_pair[i], v2_pair[j], tree_pred_pair[i][j] = 1, 1, 1
        '''
        #############################
        # FOR PROTOTYPING!!!
        # differentiable version
        tree_pred_pair_soft = self._gumbel_softmax(y_pred_mat_masked.view(-1),
                                                   self.temperature).view(M, N)  # TODO: check
        v1_pair_soft, v2_pair_soft = torch.sum(tree_pred_pair_soft, dim=1), torch.sum(
            tree_pred_pair_soft, dim=0)
        # hard-mask forward pass; soft-mask backward pass
        v1_pair = (v1_pair - v1_pair_soft).detach() + v1_pair_soft
        v2_pair = (v2_pair - v2_pair_soft).detach() + v2_pair_soft
        tree_pred_pair = (tree_pred_pair - tree_pred_pair_soft).detach() + tree_pred_pair_soft
        #############################
        '''
        # return v1_pair, v2_pair, tree_pred_pair, v1_pair_soft, v2_pair_soft, tree_pred_pair_soft, y_pred_mat_masked
        return v1_pair, v2_pair, tree_pred_pair

    def _expand_search_frontier(self, state, pair):
        v1, v2 = state.v1, state.v2
        adj_mat1, adj_mat2 = state.adj_mat1, state.adj_mat2
        y_pred_mat = state.y_pred_mat
        M, N = y_pred_mat.shape

        mask_g1 = self._expand_search_frontier_single_graph(v1, adj_mat1)
        mask_g2 = self._expand_search_frontier_single_graph(v2, adj_mat2)

        if self.loss_opt != 'None':
            assert False  # TODO: uncomment this once testing learnable MCS-RNN
            #############################
            # FOR PROTOTYPING!!!
            red_indicator1, orange_indicator1, gray_indicator1 = state.v1, mask_g1, torch.ones(
                M) - state.v1 - mask_g1
            red_indicator2, orange_indicator2, gray_indicator2 = state.v2, mask_g2, torch.ones(
                N) - state.v2 - mask_g2

            g1, g2 = pair.g1.get_nxgraph(), pair.g2.get_nxgraph()
            cvt_init_x = lambda x: torch.tensor(x, dtype=torch.float).to(FLAGS.device)
            x1, x2 = cvt_init_x(g1.init_x), cvt_init_x(
                g2.init_x)  # TODO: potentially use the prev x
            edge_list1, edge_list2 = create_edge_index(g1), create_edge_index(g2)

            sgmn_ins_1 = x1, edge_list1, red_indicator1, orange_indicator1, gray_indicator1
            sgmn_ins_2 = x2, edge_list2, red_indicator2, orange_indicator2, gray_indicator2
            out1, out2 = self.sgmn(sgmn_ins_1, sgmn_ins_2)

            state.y_pred_mat = torch.mm(out1, out2.t())  # TODO:: use matching matrix comp
            #############################

        y_pred_mat_mask = torch.ger(mask_g1, mask_g2)
        assert y_pred_mat_mask.shape == state.y_pred_mat.shape

        # Update state.
        state.frontier_mask = y_pred_mat_mask

    def _expand_search_frontier_single_graph(self, v, adj_mat):  # expand to/select neighbors
        adj_mat_t = torch.t(adj_mat * (-1 * (
                1 - v)))  # erase cols in adj then transpose # TODO: what if multiple 1s for selected nodes????? see overleaf paper
        mask_g = torch.t(-1 * v * adj_mat_t)  # select rows in adj
        # mask_g = -1*(1-mask_g)
        mask_g = (torch.sum(mask_g, dim=0) > 0).type(torch.FloatTensor).to(
            FLAGS.device)
        # mask_g = -1*(1-mask_g)
        return mask_g

    def _gumbel_softmax(self, input, temperature):
        logits = 0
        # logits = self._sample_gumbel(input.size())
        return F.softmax((input + logits) / temperature, dim=-1)  # global softmax

    # def _sample_gumbel(self, shape, eps=1e-20):
    #     U = torch.rand(shape)
    #     return -Variable(torch.log(-torch.log(U + eps) + eps)).to(FLAGS.device)

    def _get_subgraph_embedding(self, x, mask, edge_list):
        x = torch.t(mask * torch.t(x))
        x = F.relu(self.conv1(x, edge_list))
        x = torch.t(mask * torch.t(x))  # torch.t(z_pred * torch.t(x))
        x = F.relu(self.conv2(x, edge_list))
        x = torch.t(mask * torch.t(x))  # torch.t(z_pred * torch.t(x))
        x = F.relu(self.conv3(x, edge_list))
        x = torch.t(mask * torch.t(x))
        x = torch.sum(x, dim=0)
        return x

    def _update_loss(self, update_material, y_true_dict):
        v1_pair, v2_pair, tree_pred_pair, v1_pair_soft, v2_pair_soft, \
        tree_pred_pair_soft, y_pred_mat_masked = update_material
        y_pred = tree_pred_pair_soft
        # construct ground truth matrix
        y_true = torch.zeros_like(y_pred, device=FLAGS.device)
        for nid1 in y_true_dict.keys():
            nid2 = y_true_dict[nid1]
            y_true[nid1, nid2] = 1
        loss_pair = torch.sum(-1 * y_true * torch.log(y_pred + 1e-12)) \
                    / (torch.sum(y_true) + 1e-12)
        self.loss += loss_pair


def test_tree_search_find_mcs(dataset_name, log_folder):
    pred_pairs, dataset = _test_helper_load(dataset_name, log_folder)
    for (gid1, gid2), pred_pair in sorted(pred_pairs.items()):
        print(gid1, gid2, pred_pair)
        g1 = dataset.look_up_graph_by_gid(gid1)
        g2 = dataset.look_up_graph_by_gid(gid2)
        pred_pair.assign_g1_g2(g1, g2)
        true_pair_result = dataset.look_up_pair_by_gids(gid1, gid2)
        # stop_strategy = 'threshold_0.1'
        stop_strategy = 'sin_eps=0.0001_D=64'
        ts = TreeSearch(29, stop_strategy)  # TODO: change here
        result = ts.tree_search_find_mcs(pred_pair)
        print(result)
        exit(-1)


def _test_helper_load(dataset_name, log_folder):
    # Load pairwise results including node-node matching matrix,
    log_folder = join(get_model_path(), 'Our', 'logs', log_folder)
    ld = load(join(log_folder, 'final_test_pairs.klepto'))
    pairs = ld['test_data_pairs']
    print(len(pairs), 'pairs loaded')
    # Load graphs.
    dataset = load_dataset(dataset_name, 'all', 'mcs', 'bfs')  # TODO: check bfs assumption
    dataset.print_stats()
    natts, *_ = get_dataset_conf(dataset_name)
    # node_feat_name = natts[0] if len(natts) >= 1 else None  # TODO: only one node feat
    from node_feat import encode_node_features
    dataset, _ = encode_node_features(dataset)
    # TODO: should really load and reset flags but since encode_node_features only uses 'one_hot' it is fine for now
    return pairs, dataset


if __name__ == '__main__':
    dataset_name = 'aids700nef'
    log_folder = 'gmn_icml_mlp_mcs_aids700nef_2019-08-11T20-49-02.381800(plot)'
    test_tree_search_find_mcs(dataset_name, log_folder)
