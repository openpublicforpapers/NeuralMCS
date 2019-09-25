from config import FLAGS
from utils_our import get_flags_with_prefix_as_list
from eval_pairs import eval_pair_list
from eval_ranking import eval_ranking
from dataset import OurOldDataset
from collections import OrderedDict
from pprint import pprint
import numpy as np


class Eval(object):
    def __init__(self, trained_model, train_data, test_data, saver):
        self.trained_model = trained_model
        self.train_data = train_data
        self.test_data = test_data
        self.saver = saver
        self.global_result_dict = OrderedDict()

    def eval_on_test_data(self, round=None):
        if round is None:
            info = 'final_test'
            d = OrderedDict()
            self.global_result_dict[info] = d
        else:
            raise NotImplementedError()

        d['pairwise'] = self._eval_pairs(info)

        if type(self.test_data.dataset) is OurOldDataset:  # ranking
            d['ranking'] = self._eval_ranking(info)

        self.saver.save_global_eval_result_dict(self.global_result_dict)

    def _eval_pairs(self, info):
        print('Evaluating pairwise results...')
        pair_list = self.test_data.get_pairs_as_list()
        result_dict = eval_pair_list(pair_list, FLAGS)
        pprint(result_dict)
        self.saver.save_eval_result_dict(result_dict, 'pairwise')
        self.saver.save_pairs_with_results(self.test_data, self.train_data, info)
        return result_dict

    def _eval_ranking(self, info):
        print('Evaluating ranking results...')
        test_dataset = self.test_data.dataset
        gs1, gs2 = test_dataset.gs1(), test_dataset.gs2()
        m, n = len(gs1), len(gs2)
        true_ds_mat, pred_ds_mat, time_mat = \
            self._gen_ds_time_mat(gs1, gs2, m, n, test_dataset)
        result_dict, true_m, pred_m = eval_ranking(
            true_ds_mat, pred_ds_mat, FLAGS.dos_pred, time_mat)  # dos_pred!
        pprint(result_dict)
        self.saver.save_eval_result_dict(result_dict, 'ranking')
        self.saver.save_ranking_mat(true_m, pred_m, info)
        return result_dict

    def _gen_ds_time_mat(self, gs1, gs2, m, n, test_dataset):
        true_ds_mat = np.zeros((m, n))  # may need to do normalization
        pred_ds_mat = np.zeros((m, n))  # do not normalize predicted scores
        time_mat = np.zeros((m, n))
        for i, g1 in enumerate(gs1):
            for j, g2 in enumerate(gs2):
                pair = test_dataset.look_up_pair_by_gids(g1.gid(), g2.gid())
                ds_true = pair.get_ds_true(FLAGS.ds_norm, FLAGS.dos_true, FLAGS.dos_pred,
                                           FLAGS.ds_kernel)
                # TODO: optimize the above code by doing these things at matrix-level
                ds_pred = pair.get_ds_pred()
                duration = pair.get_pred_time()
                true_ds_mat[i][j] = ds_true
                pred_ds_mat[i][j] = ds_pred
                time_mat[i][j] = duration
        return true_ds_mat, pred_ds_mat, time_mat
