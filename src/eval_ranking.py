from eval_ranking_metrics import prec_at_ks, mean_reciprocal_rank, \
    mean_squared_error, mean_deviation, kendalls_tau, spearmans_rho, average_time
import numpy as np
from collections import OrderedDict


class RankingMat(object):
    """
    The RankingMat object to help evaluate the ranking result of a model.
        Terminology:
            rtn: return value of a function.
            m: # of query/testing/row graphs.
            n: # of database/training/col graphs.
    """

    def __init__(self, ds_mat, dist_or_sim, time_mat):
        self.ds_mat = ds_mat
        self.sort_id_mat = np.argsort(self.ds_mat, kind='mergesort')
        if dist_or_sim == 'sim':
            self.sort_id_mat = self.sort_id_mat[:, ::-1]
        self.time_mat = time_mat

    def m_n(self):
        return self.ds_mat.shape

    def dist_sim_mat(self):
        """
        Each RankingMat object stores either a distance matrix
            or a similarity matrix. It cannot store both.
        :return: either the distance matrix or the similairty matrix.
        """
        return self.ds_mat

    def get_time_mat(self):
        return self.time_mat

    def top_k_ids(self, qid, k, inclusive, rm=0):
        """
        :param qid: query id (0-indexed).
        :param k:
        :param inclusive: whether to be tie inclusive or not.
            For example, the ranking may look like this:
            7 (sim_score=0.99), 5 (sim_score=0.99), 10 (sim_score=0.98), ...
            If tie inclusive, the top 1 results are [7, 9].
            Therefore, the number of returned results may be larger than k.
            In summary,
                len(rtn) == k if not tie inclusive;
                len(rtn) >= k if tie inclusive.
        :return: for a query, the ids of the top k database graph
        ranked by this model.
        """
        sort_id_mat = self.sort_id_mat
        _, n = sort_id_mat.shape
        if k <= 0 or k > n:
            raise RuntimeError('Invalid k {}'.format(k))
        if not inclusive:
            return sort_id_mat[qid][:k]
        # Tie inclusive.
        dist_sim_mat = self.ds_mat
        while k < n:
            cid = sort_id_mat[qid][k - 1]
            nid = sort_id_mat[qid][k]
            if abs(dist_sim_mat[qid][cid] - dist_sim_mat[qid][nid]) <= rm:
                k += 1
            else:
                break
        return sort_id_mat[qid][:k]

    def ranking(self, qid, gid, one_based=True):
        """
        :param qid: query id (0-indexed).
        :param gid: database graph id (0-indexed) (NOT g.graph['gid']).
        :param one_based: whether to return the 1-based or 0-based rank.
            True by default.
        :return: for a query, the rank of a database graph by this model.
        """
        # Assume self is ground truth.
        sort_id_mat = self.sort_id_mat
        finds = np.where(sort_id_mat[qid] == gid)
        assert (len(finds) == 1 and len(finds[0]) == 1)
        fid = finds[0][0]
        # Tie inclusive (always when find ranking).
        dist_sim_mat = self.ds_mat
        while fid > 0:
            cid = sort_id_mat[qid][fid]
            pid = sort_id_mat[qid][fid - 1]
            if dist_sim_mat[qid][pid] == dist_sim_mat[qid][cid]:
                fid -= 1
            else:
                break
        if one_based:
            fid += 1
        return fid

    def ranking_mat(self, one_based=True):
        """
        :param one_based:
        :return: a m by n matrix representing the ranking result.
                 Note it is different from sort_id_mat.
            rtn[i][j]: For query i, the ranking of the graph j.
        """
        m, n = self.m_n()
        rtn = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                rtn[i][j] = self.ranking(i, j, one_based=one_based)
        return rtn


def eval_ranking(true_ds_mat, pred_ds_mat, dist_or_sim, time_mat):
    """
    true_ds_mat, pred_ds_mat should both mean 'sim' or 'dist'
    specified by dist_or_sim.
    In other words, if the true mat means similarity
    but the predicted mat means distance, the metrics won't work.
    """
    rtn = OrderedDict()
    assert true_ds_mat.shape == pred_ds_mat.shape == time_mat.shape
    true_m = RankingMat(true_ds_mat, dist_or_sim, time_mat)
    pred_m = RankingMat(pred_ds_mat, dist_or_sim, time_mat)
    m, n = true_ds_mat.shape
    upper = 20
    if n < upper:
        upper = n
    ks = range(1, upper + 1)
    print(ks)
    nums = prec_at_ks(true_m, pred_m, ks)
    assert len(nums) == len(ks)
    for i, k in enumerate(ks):
        rtn['prec@{}'.format(k)] = nums[i]
    rtn['mse'] = mean_squared_error(true_m, pred_m)
    rtn['rho'] = spearmans_rho(true_m, pred_m)
    rtn['tau'] = kendalls_tau(true_m, pred_m)
    rtn['dev'] = mean_deviation(true_m, pred_m)
    rtn['mrr'] = mean_reciprocal_rank(true_m, pred_m)
    rtn['time'] = average_time(pred_m)
    return rtn, true_m, pred_m