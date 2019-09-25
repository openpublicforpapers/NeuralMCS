import numpy as np
from sklearn.preprocessing import normalize

np.set_printoptions(precision=3)

D = 5
iters = 1


def main():
    # M = np.arange(1, D * D + 1).reshape(D, D)
    M = np.random.rand(D, D)
    print('M\n', M)
    S = sinkhorn(M)
    print('Done')
    # print('S\n', S)


def sinkhorn(M):
    for iter in range(iters):
        M = normalize(M, axis=1, norm='l1')  # row norm
        print('\trow\n', M)
        M = normalize(M, axis=0, norm='l1')  # col norm
        print('\tcol\n', M)
        # print('iter', iter, '\n', M)
    return M


if __name__ == '__main__':
    main()
