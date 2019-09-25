from vis_matching_v2 import plot_node_mapping
from eval_pairs import rc_thres_lsap
from dataset_config import get_dataset_conf
from load_data import load_dataset
from utils import get_model_path, get_result_path, load
from os.path import join
import numpy as np
import random
from collections import OrderedDict
from warnings import warn


def plot_true_pairs(dataset_name, num_pairs, fix_match_pos, want_gid_tuples,
                    need_eps):
    dir = join(get_result_path(), dataset_name, 'matching_vis')
    want = ['true']
    _plot_pairs(None, dataset_name, num_pairs, fix_match_pos, dir, want,
                want_gid_tuples, need_eps)


def plot_pred_pairs_from_saved_kelpto(log_folder, dataset_name, num_pairs,
                                      fix_match_pos, want_gid_tuples, need_eps,
                                      mode, pick_best):
    log_folder = join(get_model_path(), 'Our', 'logs', log_folder)
    ld = load(join(log_folder, 'final_test_pairs.klepto'))
    pairs = ld['test_data_pairs']
    print(len(pairs), 'pairs loaded')
    cnt = 0
    pairs_valid = OrderedDict()
    for _, pair in pairs.items():
        if pair.has_alignment_pred():
            cnt += 1
            pairs_valid[_] = pair
    print('{}/{}={:.2%} has pred matching matrices'.format(
        cnt, len(pairs), cnt / len(pairs)))
    pairs = pairs_valid
    dir = join(log_folder, 'matching_vis')
    want = ['true'] if pick_best else ['pred', 'true']
    _plot_pairs(pairs, dataset_name, num_pairs, fix_match_pos, dir, want,
                want_gid_tuples, need_eps, mode, pick_best)


def _plot_pairs(pairs, dataset_name, num_pairs, fix_match_pos, dir, want,
                want_gid_tuples, need_eps, mode, pick_best):
    dataset = load_dataset(dataset_name, 'all', 'mcs', 'bfs')
    dataset.print_stats()
    natts, *_ = get_dataset_conf(dataset_name)
    node_feat_name = natts[0] if len(natts) >= 1 else None  # TODO: only one node feat
    if pairs is None:
        pairs = dataset.get_all_pairs()
    pairs, num_pairs = _filter_pairs(pairs, num_pairs, want_gid_tuples)
    assert num_pairs >= 1 and len(pairs) >= num_pairs, '{} {}'.format(
        num_pairs, len(pairs))
    all_pair_gid_tuples = sorted(pairs.keys())
    random.Random(123).shuffle(all_pair_gid_tuples)
    for i in range(num_pairs):
        gid1, gid2 = all_pair_gid_tuples[i]
        # if gid1 != 106:
        #     continue
        # else:
        #     pass
        g1 = dataset.look_up_graph_by_gid(gid1).get_nxgraph()
        g2 = dataset.look_up_graph_by_gid(gid2).get_nxgraph()
        fnb = '{}_{}_{}'.format(dataset_name, g1.graph['gid'], g2.graph['gid'])
        _plot_pairs_helper(fnb, g1, g2, pairs, node_feat_name, dataset, gid1, gid2,
                           fix_match_pos, dir, want, need_eps, mode, pick_best)


def _plot_pairs_helper(fnb, g1, g2, pairs, node_feat_name, dataset, gid1, gid2,
                       fix_match_pos, dir, want, need_eps, mode, pick_best):
    pred_pair = pairs.get((gid1, gid2))
    if mode == 'tree_search':
        pred = 'tree_pred'
    else:
        pred = 'y_pred'
    score = pred_pair.match_eval_result[pred][mode]['best']['mcs_exact']
    if pick_best and score != 1.0:
        warn('Skip pair {},{} with score {}'.format(gid1, gid2, score))
        return
    for w in want:
        if w == 'true':
            pair = dataset.look_up_pair_by_gids(gid1, gid2)
            mapping = pair.get_y_true_list_dict_view()[0]
            fn = fnb + '_true'
            print(w, gid1, gid2, mapping)
            # exit(-1)
            assert (type(mapping) is dict)
            plot_node_mapping(g1, g2, mapping, node_feat_name, fix_match_pos,
                              dir, fn, need_eps, print_path=True)

        elif w == 'pred':
            # np.set_printoptions(precision=3, suppress=True)
            # yyy = pair.get_y_pred_list_mat_view(format='numpy')[0]
            # print(pair.get_y_pred_list_mat_view(format='numpy')[0])
            # exit(-1)

            if mode == 'tree_search':
                mats = pred_pair.get_tree_pred_list_mat_view(format='numpy')
            else:
                mats = pred_pair.get_y_pred_list_mat_view(format='numpy')
            for j, mat in enumerate(mats):  # handle multiple preds
                if mode == 'rc_thres_lsap':
                    mat = rc_thres_lsap(mat, 0.5)  # TODO: hacky; other modes; threshold

                elif mode == 'tree_search':
                    pass
                else:
                    raise NotImplementedError()
                # print(mat)
                mapping = _turn_matching_mat_to_mapping_dict(mat)
                fn = fnb + '_pred_' + str(j) + '_' + \
                     str(pred_pair.match_eval_result[pred][mode]['best']['mcs_exact'])
                print(w, gid1, gid2, mapping)
                # exit(-1)
                assert (type(mapping) is dict)
                plot_node_mapping(g1, g2, mapping, node_feat_name, fix_match_pos,
                                  dir, fn, need_eps, print_path=True)
        else:
            assert False


def _filter_pairs(pairs, num_pairs, want_gid_tuples):
    if want_gid_tuples:
        new_pairs = {}
        for gid_tuple, pair in pairs.items():
            if gid_tuple in want_gid_tuples:
                new_pairs[gid_tuple] = pair
        new_num_pairs = len(new_pairs)
    else:
        new_pairs = pairs
        new_num_pairs = num_pairs
    return new_pairs, new_num_pairs


def _turn_matching_mat_to_mapping_dict(mat):
    # TODO: make sure y_pred_mat consists of only 0s and 1s
    indx, indy = np.where(mat == 1)
    rtn = {}
    for i, x in enumerate(indx):
        rtn[x] = indy[i]
    return rtn


if __name__ == '__main__':
    # dataset_name = 'aids700nef'
    # num_pairs = 10
    # fix_match_pos = False
    # want_gid_tuples = [(6786, 2012)]
    # plot_true_pairs(dataset_name, num_pairs, fix_match_pos, want_gid_tuples)

    dataset_name = 'redditmulti10k'
    num_pairs = 100  # only used if want_gid_tuples is empty
    log_folder = 'GMN_icml_mlp-OurLoss-tree_search_redditmulti10k_2019-09-19T18-33-05.786552'
    fix_match_pos = False
    # mode = 'rc_thres_lsap'
    mode = 'tree_search'
    want_gid_tuples = []  # if empty, use num_pairs
    need_eps = False
    pick_best = True
    plot_pred_pairs_from_saved_kelpto(log_folder, dataset_name, num_pairs,
                                      fix_match_pos, want_gid_tuples, need_eps,
                                      mode, pick_best)
