from utils import get_result_path
import pandas as pd
import os
import copy
from ast import literal_eval
from os.path import join

name = 'aids700nef'
dataset = join(get_result_path(), name, 'mcs',
               'mcs_aids700nef_mccreesh2017_2018-11-27T02:36:27.553945_redacted-desktop_all_4cpus')
df = pd.read_csv('{}.csv'.format(dataset), sep=',')
# for index, chunk in enumerate(pd.read_csv('{}.csv'.format(dataset), sep=',', chunksize=1)):
print('read csv')
hits = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
cur_hit = 0
for index, row in df.iterrows():
    perc = index / len(df)
    if cur_hit < len(hits) and abs(perc - hits[cur_hit]) <= 0.05:
        print('{}/{}={:.1%}'.format(index, len(df), perc))
        cur_hit += 1

    node_mapping = {}
    edge_mapping = row['node_mapping']

    edge_mapping = literal_eval(edge_mapping)[0]
    row['edge_mapping'] = [copy.deepcopy(edge_mapping)]

    # one node with one node mapping
    if edge_mapping == {}:
        pass
    # two nodes with two nodes mapping
    elif len(edge_mapping) == 1:
        nodes_i, nodes_j = edge_mapping.popitem()
        node_mapping[nodes_i[0]] = nodes_j[0]
        node_mapping[nodes_i[1]] = nodes_j[1]

    # O.W.
    else:
        for nodes_i, nodes_j in edge_mapping.items():
            for node_i in nodes_i:
                if node_mapping.get(node_i) is None:
                    node_mapping[node_i] = list(nodes_j)
                else:
                    # if node_i == '6':
                    #     kk = node_mapping[node_i]
                    #     a = set(node_mapping[node_i])
                    #     b = set(nodes_j)
                    #     c = b.intersection(a)
                    #     d = list(c)
                    node_mapping[node_i] = list(set(nodes_j).intersection(set(node_mapping[node_i])))[0] if type(
                        node_mapping[node_i]) == list else list(set(nodes_j).intersection(set([node_mapping[node_i]])))[
                        0]

        for n_1, n_2 in node_mapping.items():
            if type(n_2) == list:
                n_1_p, n_2_p = None, None
                for nodes_i, nodes_j in edge_mapping.items():
                    if n_1 in nodes_i:
                        l = list(nodes_i)
                        l.remove(n_1)
                        n_1_p = l[0]
                        assert (type(node_mapping[n_1_p]) != list)
                        n_2_p = node_mapping[n_1_p]
                assert (n_1_p is not None and n_2_p is not None)
                n_2 = list(n_2)
                n_2.remove(n_2_p)
                assert (len(n_2) == 1)
                node_mapping[n_1] = n_2[0]
                assert (type(node_mapping[n_1]) != list)

    row['node_mapping'] = [node_mapping]

    # chunk.to_csv('{}.csv'.format(dataset[:-5]), header=False, mode='a', chunksize=1)
    row.to_csv('{}_preproc.csv'.format(dataset[:-5]), header=False, mode='a', chunksize=1)
