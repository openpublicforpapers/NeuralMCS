import numpy as np
from scipy.stats import hmean, kendalltau, spearmanr


def prec_at_ks(true_m, pred_m, ks):
    """
    Ranking-based. prec@ks.
    :param true_m: result object indicating the ground truth.
    :param pred_m: result object indicating the prediction.
    :param ks: 1-based integer id list.
    :return: list of floats indicating the average precision at different ks.
    """
    m, n = true_m.m_n()
    assert (true_m.m_n() == pred_m.m_n())
    ps = np.zeros((m, len(ks)))
    for i in range(m):
        for k_idx, k in enumerate(ks):
            assert (type(k) is int and k > 0 and k <= n)
            true_ids = true_m.top_k_ids(i, k, inclusive=True, rm=0)
            pred_ids = pred_m.top_k_ids(i, k, inclusive=True, rm=0)
            ps[i][k_idx] = \
                min(len(set(true_ids).intersection(set(pred_ids))), k) / k
    return np.mean(ps, axis=0)


def mean_reciprocal_rank(true_m, pred_m):
    """
    Ranking based. MRR.
    :param true_m: result object indicating the ground truth.
    :param pred_m: result object indicating the prediction.
    :return: float indicating the mean reciprocal rank.
    """
    m, n = true_m.m_n()
    assert (true_m.m_n() == pred_m.m_n())
    topanswer_ranks = np.zeros(m)
    for i in range(m):
        # There may be multiple graphs with the same dist/sim scores
        # as the top answer by the true_m model.
        # Select one with the lowest (minimum) rank
        # predicted by the pred_m model for mrr calculation.
        true_ids = true_m.top_k_ids(i, 1, inclusive=True, rm=0)
        assert (len(true_ids) >= 1)
        min_rank = float('inf')
        for true_id in true_ids:
            pred_mank = pred_m.ranking(i, true_id, one_based=True)
            min_rank = min(min_rank, pred_mank)
        topanswer_ranks[i] = min_rank
    return 1.0 / hmean(topanswer_ranks)


def mean_squared_error(true_m, pred_m):
    """
    Regression-based. L2 difference between the ground-truth sim/dist
        and the predicted sim/dist.
    :return:
    """
    assert (true_m.m_n() == pred_m.m_n())
    A = true_m.dist_sim_mat()
    # A = _proc_true_ds_mat(A, true_m, pred_m, ds_kernel)
    B = pred_m.dist_sim_mat()
    return ((A - B) ** 2).mean() / 2


def mean_deviation(true_m, pred_m):
    """
    Regression-based. L1 difference between the ground-truth sim/dist
        and the predicted sim/dist.
    :return:
    """
    assert (true_m.m_n() == pred_m.m_n())
    A = true_m.dist_sim_mat()
    # A = _proc_true_ds_mat(A, true_m, pred_m, ds_kernel)
    B = pred_m.dist_sim_mat()
    return np.abs(A - B).mean()


def average_time(r):
    return np.mean(r.get_time_mat())


def kendalls_tau(true_m, pred_m):
    return _ranking_metric(kendalltau, true_m, pred_m)


def spearmans_rho(true_m, pred_m):
    return _ranking_metric(spearmanr, true_m, pred_m)


def _ranking_metric(ranking_metric_func, true_m, pred_m):
    y_true = true_m.ranking_mat()
    y_scores = pred_m.ranking_mat()
    scores = []
    m, n = true_m.m_n()
    assert (true_m.m_n() == pred_m.m_n())
    for i in range(m):
        scores.append(ranking_metric_func(y_true[i], y_scores[i])[0])
    return np.mean(scores)


if __name__ == '__main__':
    y_true = np.array([1, 2, 3, 4, 5])
    y_scores = np.array([1, 1, 1, 1, 1])
    kendalltau = kendalltau(y_true, y_scores)
    print(kendalltau)
