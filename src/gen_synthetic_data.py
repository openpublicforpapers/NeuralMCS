from graph_pair import GraphPair
import networkx as nx


def gen_synthetic_mcs(gs):
    for g in gs:
        assert type(g) is nx.Graph
    rtn = []
    for g in gs:
        g_fake = g  # TODO: pertub
        rtn.append(GraphPair(g, g_fake, [{0: 0, 2: 1, 1: 2}]))
    return rtn
    # TODO for Amlan


def test_gen_synthetic_mcs():
    pass

if __name__ == '__main__':
    test_gen_synthetic_mcs()
