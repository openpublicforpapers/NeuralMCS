from scipy.optimize import linear_sum_assignment


def max_emb_sim(y_pred_mat, y_true_mat, pair):
    true_left, true_right = _gen_node_embs(y_true_mat, pair)
    pred_left, pred_right = _gen_node_embs(y_pred_mat, pair)
    sim_left = _wasserstein_sim_mne(true_left, pred_left)
    sim_right = _wasserstein_sim_mne(true_right, pred_right)
    return sim_left, sim_right


def _gen_node_embs(y_mat, pair):
    x_left, x_right = pair.get_xs()  # TODO: may need to handle node embeddings at different GCN layers
    # y_mat[0][0] = 1  # TODO: comment
    indices_left = y_mat.any(axis=1)
    indices_right = y_mat.any(axis=0)
    x_left = x_left[indices_left == 1, :]
    x_right = x_right[indices_right == 1, :]
    return x_left, x_right


def _wasserstein_sim_mne(x1, x2):
    mne = x1.dot(x2.T)
    row_ind, col_ind = linear_sum_assignment(-mne)  # TODO: check the negative sign
    return mne[row_ind, col_ind].sum()
