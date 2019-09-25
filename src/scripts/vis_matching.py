from utils import get_temp_path
import random
from collections import OrderedDict
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
import plotly.io as pio
import plotly.graph_objs as go
import os
import time
from os.path import expanduser, join

home = expanduser('~')
orca_path = join(home, 'anaconda3', 'bin', 'orca')
print(orca_path)
pio.orca.config.executable = orca_path
# pio.orca.config.executable = '/usr/local/bin/orca'
result_folder = get_temp_path()
graph_name = 'MCS'


def vis_matching(g1, g2, nn_map):
    pos1 = graphviz_layout(g1)
    pos2 = graphviz_layout(g2)

    # xpos of right most node in g1 and the left most node in g2
    max_g1_x = max([pos[0] for pos in pos1.values()])
    min_g2_x = min([pos[0] for pos in pos2.values()])
    diff = abs(max_g1_x - min_g2_x) * 1.5 / 2

    # move g1 right and g2 left
    for k, v in pos1.items():
        pos1[k] = (v[0] - diff, v[1])
    for k, v in pos2.items():
        pos2[k] = (v[0] + diff, v[1])

    nx.set_node_attributes(g1, pos1, 'pos')
    nx.set_node_attributes(g2, pos2, 'pos')
    # draw edges
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1, color='#000'),
        hoverinfo='none',
        mode='lines')

    edge_connected_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.7, color='#FF0000', dash='dash'),
        hoverinfo='none',
        mode='lines'
    )

    for edge in nn_map.items():
        if edge[1] == -1:
            break
        x0, y0 = g1.node[edge[0]]['pos']
        x1, y1 = g2.node[edge[1]]['pos']
        edge_connected_trace['x'] += tuple([x0, x1, None])
        edge_connected_trace['y'] += tuple([y0, y1, None])

    for edge in g1.edges():
        x0, y0 = g1.node[edge[0]]['pos']
        x1, y1 = g1.node[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    for edge in g2.edges():
        x0, y0 = g2.node[edge[0]]['pos']
        x1, y1 = g2.node[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])


    g1_node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        textposition='bottom center',
        marker=dict(
            color='#00a9ff',
            size=10,
        ),
        line=dict(width=2))

    g2_node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        textposition='bottom center',
        marker=dict(
            color='#774ca4',
            size=10,
        ),
        line=dict(width=2))

    for node in g1.nodes():
        x, y = g1.node[node]['pos']
        g1_node_trace['x'] += tuple([x])
        g1_node_trace['y'] += tuple([y])
        g1_node_trace['text'] += tuple(str(node) if nn_map[node] != -1 else '')


    for node in g2.nodes():
        x, y = g2.node[node]['pos']
        g2_node_trace['x'] += tuple([x])
        g2_node_trace['y'] += tuple([y])
        g2_node_trace['text'] += tuple(str(node) if node in nn_map.values() else '')

    fig = go.Figure(data=[edge_trace, edge_connected_trace, g1_node_trace, g2_node_trace],
                    layout=go.Layout(
                        title='MCS G1 and G2',
                        titlefont=dict(size=16),
                        hovermode='closest',
                        showlegend=False,
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    timestr = time.strftime("%Y-%m-%d-%H:%M:%S")
    fname = "{}_{}.png".format(graph_name, timestr)
    result_path = os.path.join(result_folder,fname)
    pio.write_image(fig, result_path)


def test_vis_matching():
    g1 = nx.fast_gnp_random_graph(12, 0.2, seed=123)
    g2 = nx.fast_gnp_random_graph(5, 0.2, seed=456)
    nn_map = OrderedDict()
    nl = list(g2.nodes())
    random.Random(123).shuffle(nl)
    if g1.number_of_edges() < g2.number_of_edges():
        g1, g2 = g2, g1  # make sure g1 has extra nodes
    for i, n1 in enumerate(sorted(g1.nodes())):
        if i < len(nl):
            nn_map[n1] = nl[i]
        else:
            nn_map[n1] = -1  # extra nodes of g1 are mapped to nothing
    print(nn_map)
    vis_matching(g1, g2, nn_map)


if __name__ == '__main__':
    test_vis_matching()
