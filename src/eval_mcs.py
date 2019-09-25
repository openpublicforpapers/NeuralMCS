from dataset_config import get_dataset_conf
from utils import timeout
import networkx.algorithms.isomorphism as iso
import networkx as nx
import numpy as np
from warnings import warn

ISOMORPHISM_TIMEOUT = 5  # time out after 5 seconds (use heuristic checking)


def mcs_size(y_pred_mat, y_true_mat, pair, FLAGS):
    # TODO: make sure y_pred_mat consists of only 0s and 1s
    true_left, true_right, true_gid1, true_gid2 = _gen_nx_subgraph(y_true_mat, pair)
    pred_left, pred_right, pred_gid1, pred_gid2 = _gen_nx_subgraph(y_pred_mat, pair)

    # Only take the largest connected component
    xtract = lambda gs: [max(nx.connected_component_subgraphs(g), key=len)
                         if g.number_of_nodes() > 0 else g for g in gs]
    true_left, true_right, pred_left, pred_right = xtract(
        [true_left, true_right, pred_left, pred_right])

    natts, eatts, *_ = get_dataset_conf(FLAGS.dataset)
    nm = iso.categorical_node_match(
        natts, [''] * len(natts))  # TODO: check the meaning of default value
    em = iso.categorical_edge_match(
        eatts, [''] * len(eatts))  # TODO: check the meaning of default value

    # Check to make sure the ground-truth MCS is isomorphic.
    if not FLAGS.debug:  # check the ground-truth only for non-debugging datasets
        is_iso, iso_exact = _is_isomorphic_with_timeout(
            true_left, true_right, node_match=nm, edge_match=em)
        _warn_non_exact_iso(iso_exact, pair, 'ground-truth')
        assert is_iso, \
            'Even the ground-truth subgraphs are not isomorphic! Pair {},{}'. \
                format(true_gid1, true_gid2)
    is_iso, iso_exact = _is_isomorphic_with_timeout(
        pred_left, pred_right, node_match=nm, edge_match=em)
    _warn_non_exact_iso(iso_exact, pair, 'pred')
    pn1, pn2, tn = pred_left.number_of_nodes(), pred_right.number_of_nodes(), \
                   true_left.number_of_nodes()
    if not is_iso:  # the predicted subgraphs are not even isomorphic
        # iso or not; fraction; deviation; exact or not
        return 0, 0, abs((pn1 + pn2) / 2 - tn), 0  # exact or not
    pn, tn = pred_left.number_of_nodes(), true_left.number_of_nodes()
    if tn == 0:  # true MCS is empty
        if pn == 0:
            # iso or not; fraction; deviation; exact or not
            return 1, 1, 0, 1
        else:  # true MCS solver counts # of edges in MCS, so if only one node, tn == 0
            assert is_iso  # predicted subgraphs are isomorphic
            return 1, 1, 0, 1
    assert pn1 == pn2  # b.c. iso
    pn = pn1
    frac = pn / tn
    if not FLAGS.debug and frac > 1:
        '''
        import matplotlib.pyplot as plt
        plt.subplot(121)
        nx.draw(pred_left)
        plt.subplot(122)
        nx.draw(pred_right)
        plt.show()
        raise ValueError('The predicted MCS is even larger than '
                         'the ground-truth MCS size pn={} tn={} '
                         'for pair {}, {}'.format(pn, tn, true_gid1, true_gid2))
        '''
        print('The predicted MCS is even larger than '
                         'the ground-truth MCS size pn={} tn={} '
                         'for pair {}, {}'.format(pn, tn, true_gid1, true_gid2))
    if frac > 1:
        frac = 0
    # iso or not; fraction; deviation; exact or not
    return 1, frac, abs(pn - tn), 1 if pn == tn else 0


def _is_isomorphic_with_timeout(g1, g2, node_match, edge_match):
    try:
        with timeout(seconds=ISOMORPHISM_TIMEOUT):
            result = nx.is_isomorphic(
                g1, g2, node_match=node_match, edge_match=edge_match)
            return result, True
    except TimeoutError:
        result = nx.could_be_isomorphic(g1, g2)
        return result, False


def _warn_non_exact_iso(iso_exact, pair, info):
    if not iso_exact:
        g1, g2 = pair.get_g1_g2()
        warn('\n{} Pair ({},{}) with #nodes=({},{}) has non-exact iso checking '
             'after {} seconds'.
             format(info, g1.gid(), g2.gid(),
                    g1.get_nxgraph().number_of_nodes(),
                    g2.get_nxgraph().number_of_nodes(), ISOMORPHISM_TIMEOUT))


def _gen_nx_subgraph(y_mat, pair):
    # y_mat[0][1] = 1
    indices_left = _gen_nids(y_mat, 1)
    indices_right = _gen_nids(y_mat, 0)
    g1, g2 = pair.g1.get_nxgraph(), pair.g2.get_nxgraph()
    g1_subgraph = g1.subgraph(indices_left)  # g1 is left
    g2_subgraph = g2.subgraph(indices_right)  # g2 is right
    # print(indices_left, '@@@\n', indices_right)
    return g1_subgraph, g2_subgraph, g1.graph['gid'], g2.graph['gid']


def _gen_nids(y_mat, axis):
    if axis == 0:
        rtn = np.where(y_mat == 1)[1]
    elif axis == 1:
        rtn = np.where(y_mat.T == 1)[1]
    else:
        assert False
    return list(rtn)


def _turn_matching_mat_to_mapping_dict(mat):
    # TODO: make sure y_pred_mat consists of only 0s and 1s
    indx, indy = np.where(mat == 1)
    rtn = {}
    for i, x in enumerate(indx):
        rtn[x] = indy[i]
    return rtn


if __name__ == '__main__':
    from load_data import load_dataset

    dataset_name = 'aids700nef'
    dataset = load_dataset(dataset_name, 'all', 'mcs')

    import os

    num_is_iso = 0
    num_not_iso = 0

    for graph_file in os.listdir('../data/' + dataset_name + '/mappings'):
        [gid1, gid2] = (graph_file.split('.')[0]).split('_')
        gid1, gid2 = int(gid1), int(gid2)

        pair = dataset.look_up_pair_by_gids(gid1, gid2)
        pair.assign_g1_g2(dataset.look_up_graph_by_gid(gid1), dataset.look_up_graph_by_gid(gid2))
        y_true_mat_list = pair.get_y_true_list_mat_view(format='numpy')
        true_left, true_right, true_gid1, true_gid2 = _gen_nx_subgraph(y_true_mat_list[0], pair)
        natts, eatts, *_ = get_dataset_conf(dataset_name)
        nm = iso.categorical_node_match(natts, [''] * len(
            natts))  # TODO: check the meaning of default value
        em = iso.categorical_edge_match(eatts, [''] * len(
            eatts))  # TODO: check the meaning of default value
        is_iso = nx.is_isomorphic(true_left, true_right, node_match=nm, edge_match=em)

        if not is_iso:
            num_not_iso += 1
            print('gid {},{} is not isomorphic!'.format(str(gid1), str(gid2)))
        else:
            num_is_iso += 1

    print('found {} isomorphic mcs'.format(num_is_iso))
    print('found {} non-isomorphic mcs'.format(num_not_iso))
    '''
    gid1=11074
    gid2=11075

    pair = dataset.look_up_pair_by_gids(gid1, gid2)
    pair.assign_g1_g2(dataset.look_up_graph_by_gid(gid1), dataset.look_up_graph_by_gid(gid2))
    y_true_mat_list = pair.get_y_true_list_mat_view(format='numpy)
    true_left, true_right, true_gid1, true_gid2 = _gen_nx_subgraph(y_true_mat_list[0], pair)
    natts, eatts, *_ = get_dataset_conf(dataset_name)
    nm = iso.categorical_node_match(natts, [''] * len(natts))  # TODO: check the meaning of default value
    em = iso.categorical_edge_match(eatts, [''] * len(eatts))  # TODO: check the meaning of default value
    is_iso = nx.is_isomorphic(true_left, true_right, node_match=nm, edge_match=em)

    import matplotlib.pyplot as plt
    plt.subplot(221)
    nx.draw(pair.g1.get_nxgraph(), node_size=1)
    plt.subplot(222)
    nx.draw(pair.g2.get_nxgraph(), node_size=1)

    plt.subplot(223)
    nx.draw(true_left, node_size=1)
    plt.subplot(224)
    nx.draw(true_right, node_size=1)

    print(true_left.number_of_nodes())
    print(true_right.number_of_nodes())
    print(is_iso)
    plt.show()
    '''
