from config import FLAGS
import torch
import torch.nn as nn
import networkx as nx
import numpy as np

class Affinity(nn.Module):
    def __init__(self, F, U):
        super(Affinity, self).__init__()
        self.F = F
        self.U = U
        self.start = True
        self.lamda1, self.lamda2 = torch.empty(128, 128), torch.empty(128, 128)
        self.cuda = 'cuda' in FLAGS.device

    def forward(self, ins, batch_data, model):
        # 1a. Get data
        F_layer = _get_layer_i(model, self.F - 1)  # 1-based to 0-based
        F = model.get_layer_output(F_layer)

        U_layer = _get_layer_i(model, self.U - 1)  # 1-based to 0-based
        U = model.get_layer_output(U_layer)

        batch_data.split_into_pair_list(F, 'F')
        pair_list = batch_data.split_into_pair_list(U, 'U')
        batch_data.split_into_pair_list(U, 'x')

        # 1b. Initialize Lambda parameter if not initialized and form it
        if self.start:
            # Initialize with the size returned by the previous layer
            self.start = False
            d = F.shape[1]  # Embedding size for layer F
            self.lamda1 = torch.eye(d)
            self.lamda2 = torch.eye(d)

        lamda_m = torch.cat((self.lamda1, self.lamda2), dim=-1)
        lamda = torch.cat((lamda_m, lamda_m), dim=-2)
        lamda = torch.nn.ReLU()(lamda)

        """
        Lambda is a concatenation shown as follows
        lamda = [lamda_1  lamda_2
                 lamda_2  lamda_1]

        """

        out_list = []

        # Iterate Per graph
        for gp in pair_list:
            g1 = gp.g1
            g2 = gp.g2

            # o. Get sorted Nodes
            nodes1 = list(g1.nxgraph)
            nodes2 = list(g2.nxgraph)
            nodes1.sort()
            nodes2.sort()

            # i. Recover incidence matrices (Note H == G for undirected)
            H1 = nx.incidence_matrix(g1.nxgraph, nodes1).tocoo()
            H2 = nx.incidence_matrix(g2.nxgraph, nodes2).tocoo()

            # ib. Densify Incidence
            # Note. Tried to sparsify but torch was getting some errors.
            H1D = torch.tensor(H1.todense())
            H2D = torch.tensor(H2.todense())

            # ii. Get edge matrices F and node matrices U (output from the previous GCN layer)
            F1 = g1.F
            F2 = g2.F

            U1 = g1.U
            U2 = g2.U

            # iii. Build X and Y edge descriptors
            if self.cuda:
                X0 = (F1[H1.row]).cpu()
                Y0 = (F2[H2.row]).cpu()
            else:
                X0 = F1[H1.row]
                Y0 = F2[H2.row]

            X = torch.cat((X0[0::2], X0[1::2]), 1)
            Y = torch.cat((Y0[0::2], Y0[1::2]), 1)

            # iv. Compute Me and Mp matrix
            matTmp = lamda.matmul(Y.transpose(0, 1))
            Me = X.matmul(matTmp)
            Mp = U1.matmul(U2.transpose(0, 1))

            # v. Destroy your CPU and perform M computation and Kronecker products
            # Note. tried sparsifying but torch gave backprop errors
            if self.cuda:
                diag_Mp = diagonalize(Mp.cpu())
                diag_Me = diagonalize(Me.cpu())
            else:
                diag_Mp = diagonalize(Mp)
                diag_Me = diagonalize(Me)

            kron_H = kronecker(H2D, H1D)
            """
            Note. Computing the kronecker can generate `RuntimeError` errors:
                Train error: [enforce fail at CPUAllocator.cpp:56] posix_memalign(&data, gAlignment, nbytes) == 0. 12 vs 0

            To fix this I put a try-catch on the TRAIN and TEST functions in train.py

            The error can occur in Forward pass OR backprop, so can't put a try-catch in here
            Don't know what to do, this is simply not scalable because Kroenecker does
            """

            # TODO comment this. ONLY FOR TESTING FAILURES
            # if np.random.uniform() < 0.5:
            # 	raise RuntimeError("TEST")

            Kr = torch.matmul(diag_Me, kron_H.transpose(0, 1))
            Kr_a = kron_H @ Kr
            M = Kr_a + diag_Mp
            out_list.append(M)

        return out_list


def kronecker(matrix1, matrix2):
    tmp = torch.ger(matrix1.view(-1), matrix2.view(-1)).reshape(
        *(matrix1.size() + matrix2.size())).permute(
        (0, 2, 1, 3)).reshape(matrix1.size(0) * matrix2.size(0),
                              matrix1.size(1) * matrix2.size(1))
    return tmp.type("torch.FloatTensor")


def diagonalize(tensor, dense=True):
    n = tensor.numel()
    if dense:
        return torch.diag(tensor.view((n,)))
    else:
        npr = np.arange(0, n)
        ra = np.stack([npr, npr], axis=1).T
        t_range = torch.from_numpy(ra).type("torch.LongTensor")
        sparse = torch.sparse_coo_tensor(t_range, tensor.view((n,)),
                                         requires_grad=False)
        return sparse


class PowerIteration(nn.Module):
    def __init__(self, k):
        super(PowerIteration, self).__init__()
        self.k = k

    def forward(self, ins, batch_data, model):
        out_list = []
        for matrix_tensor in ins:
            # Start with a 1s vector
            # Multiply it by the ins tensor, then normalize it by the matrix
            # Keep multiplying the same vector `k` times and then normalizing

            v = torch.ones(matrix_tensor.shape[0])

            for i in range(0, self.k):
                v = matrix_tensor.matmul(v)
                v = v / torch.norm(v)
            out_list.append(v)

        return out_list


class BiStochastic(nn.Module):
    def __init__(self, k):
        super(BiStochastic, self).__init__()
        self.k = k
        self.cuda = 'cuda' in FLAGS.device

    def forward(self, ins, batch_data, model):
        """
            This layer reshapes the PowerIteration vector into a NxM matrix.
            Then normalize row-wise and column-wise alternating (odd and even iteration) `k` times.

            Basically, divide each row by the sum of that row
            Then divide each column by the sum of that column

        """
        out_list = []
        for i in range(len(batch_data.pair_list)):
            n = batch_data.pair_list[i].n
            m = batch_data.pair_list[i].m

            S = ins[i].view(n, m)

            for j in range(1, self.k + 1):
                if j % 2:
                    S = S / torch.sum(S, dim=(0,))
                else:
                    S = torch.t(torch.t(S) / torch.sum(S, dim=(1,)))
                # Tried to use a division without transposing, but Torch doesn't seem to have it

            if self.cuda:
                transPoseS = torch.t(S).cuda()
            else:
                transPoseS = torch.t(S)

            out_list.append(transPoseS)
            batch_data.pair_list[i].assign_y_pred_list([transPoseS for _ in range(FLAGS.n_outputs)],
                                                       format='torch_{}'.format(FLAGS.device))
        # batch_data.pair_list[i].assign_ds_pred(transPoseS)
        # model.store_layer_output(self, out_list)

        return out_list


def _get_layer_i(model, i):
    assert i >= 0
    return model.layers[i]