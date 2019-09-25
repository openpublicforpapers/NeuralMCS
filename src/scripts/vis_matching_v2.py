from utils import append_ext_to_filepath, create_dir_if_not_exists
import matplotlib

# Fix font type for ACM paper submission.
# matplotlib.use('Agg')
matplotlib.rc('font', **{'family': 'serif', 'size': 0})  # turn off tick labels
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
from collections import OrderedDict, defaultdict
from os.path import join, dirname
from warnings import warn

TYPE_COLOR_MAP = {
    'C': '#ff6666',
    'O': 'lightskyblue',
    'N': 'yellowgreen',
    'S': 'yellow',
    'movie': '#ff6666',
    'tvSeries': '#ff6666',
    'actor': 'lightskyblue',
    'actress': '#ffb3e6',
    'director': 'yellowgreen',
    'composer': '#c2c2f0',
    'producer': '#ffcc99',
    'cinematographer': 'gold'}

FAVORITE_COLORS = ['#ff6666', 'lightskyblue', 'yellowgreen', '#c2c2f0', 'gold',
                   '#ffb3e6', '#ffcc99', '#E0FFFF', '#7FFFD4', '#20B2AA',
                   '#FF8C00', '#ff1493',
                   '#FFE4B5', '#e6e6fa', '#7CFC00']


def plot_node_mapping(g1, g2, mapping, node_feat_name, fix_match_pos,
                      dir, fn, need_eps, print_path):
    assert type(mapping) is dict
    g1, feat_dict_1 = _gen_feat_dict(g1, node_feat_name)
    g2, feat_dict_2 = _gen_feat_dict(g2, node_feat_name)

    pos_g1 = _sorted_dict(graphviz_layout(g1, prog='neato'))
    pos_g2 = _sorted_dict(graphviz_layout(g2, prog='neato'))

    _orig(g1, pos_g1, feat_dict_1, g2, pos_g2, feat_dict_2, node_feat_name,
          dir, fn, need_eps, print_path)

    _blue_red(mapping, g1, pos_g1, feat_dict_1, g2, pos_g2, feat_dict_2,
              fix_match_pos, dir, fn, need_eps, print_path)

    _detail(mapping, g1, pos_g1, feat_dict_1, g2, pos_g2, feat_dict_2,
            fix_match_pos, dir, fn, need_eps, print_path)

    _paper_style(mapping, node_feat_name, g1, pos_g1, g2, pos_g2, dir, fn,
                 need_eps, print_path)


def _orig(g1, pos_g1, feat_dict_1, g2, pos_g2, feat_dict_2, node_feat_name,
          dir, fn, need_eps, print_path):
    ntypes = defaultdict(int)
    if node_feat_name is not None:
        for node, ndata in g1.nodes(data=True):
            ntypes[ndata[node_feat_name]] += 1
        for node, ndata in g2.nodes(data=True):
            ntypes[ndata[node_feat_name]] += 1
    color_map = _gen_color_map(ntypes)

    color_g1 = _gen_orig_node_colors(g1, node_feat_name, color_map)
    color_g2 = _gen_orig_node_colors(g2, node_feat_name, color_map)

    _plot(g1, color_g1, pos_g1, feat_dict_1, g2, color_g2, pos_g2, feat_dict_2,
          dir, fn + '_orig', need_eps, print_path)


def _blue_red(mapping, g1, pos_g1, feat_dict_1, g2, pos_g2, feat_dict_2,
              fix_match_pos, dir, fn, need_eps, print_path):
    color_g1 = []
    color_g2 = []

    for node in range(g1.number_of_nodes()):
        color_g1.append('lightskyblue')
    for node in range(g2.number_of_nodes()):
        color_g2.append('lightskyblue')

    for node in mapping.keys():
        if fix_match_pos:
            pos_g2[mapping[node]] = pos_g1[node]  # matched nodes are in the same position
        color_g1[_get_node(g1.nodes, node)] = 'coral'
        color_g2[_get_node(g2.nodes, mapping[node])] = 'coral'

    _plot(g1, color_g1, pos_g1, feat_dict_1, g2, color_g2, pos_g2, feat_dict_2,
          dir, fn + '_blue_red', need_eps, print_path)


def _paper_style(mapping, node_feat_name, g1, pos_g1, g2, pos_g2, dir, fn,
                 need_eps, print_path):
    color_g1 = []
    color_g2 = []

    for node in range(g1.number_of_nodes()):
        color_g1.append('lightgray')
    for node in range(g2.number_of_nodes()):
        color_g2.append('lightgray')

    _, feat_dict_1 = _gen_feat_dict(g1, node_feat_name, False)
    _, feat_dict_2 = _gen_feat_dict(g2, node_feat_name, False)
    for fix_node_pos in [False, True]: # first False, then True so that updates later
        for node in mapping.keys():
            color_g1[_get_node(g1.nodes, node)] = 'coral'
            color_g2[_get_node(g2.nodes, mapping[node])] = 'coral'
            if fix_node_pos:
                pos_g2[mapping[node]] = pos_g1[node]  # matched nodes are in the same position

        _plot(g1, color_g1, pos_g1, feat_dict_1, g2, color_g2, pos_g2, feat_dict_2,
              dir, fn + '_paper_style{}'.format('_fix' if fix_node_pos else ''),
              need_eps, print_path)


def _detail(mapping, g1, pos_g1, feat_dict_1, g2, pos_g2, feat_dict_2,
            fix_match_pos, dir, fn, need_eps, print_path):
    color_g1 = []
    color_g2 = []

    for node in range(g1.number_of_nodes()):
        color_g1.append('lightgray')
    for node in range(g2.number_of_nodes()):
        color_g2.append('lightgray')

    ntypes = defaultdict(int)
    for node in mapping.keys():
        ntypes[node] += 1
    color_map = _gen_color_map(ntypes)

    for node in mapping.keys():
        if fix_match_pos:
            pos_g2[mapping[node]] = pos_g1[node]
        color_g1[_get_node(g1.nodes, node)] = color_map[node]
        color_g2[_get_node(g2.nodes, mapping[node])] = color_map[node]

    _plot(g1, color_g1, pos_g1, feat_dict_1, g2, color_g2, pos_g2, feat_dict_2,
          dir, fn + '_detail', need_eps, print_path)


def _get_node(node_mapping, node):
    for i in range(len(node_mapping)):
        if list(node_mapping)[i] == node:
            return i
    # assert False


def _gen_feat_dict(g, node_feat_name, need_node_id=True):
    feat_dict = {}
    node_mapping = {}
    for node in range(g.number_of_nodes()):
        node_mapping[node] = node
        if need_node_id:
            feat = '{}'.format(node)
            if node_feat_name is not None:
                feat += '_{}'.format(g.nodes[node][node_feat_name])
        else:
            feat = ''
            if node_feat_name is not None:
                feat += '{}'.format(g.nodes[node][node_feat_name])
        feat_dict[node] = feat
    g = nx.relabel_nodes(g, node_mapping)
    return g, _sorted_dict(feat_dict)


def _gen_orig_node_colors(g, node_label_name, color_map):
    if node_label_name is not None:
        color_values = []
        node_color_labels = _sorted_dict(nx.get_node_attributes(g, node_label_name))
        for node_label in node_color_labels.values():
            color = TYPE_COLOR_MAP.get(node_label, None)
            if color is None:
                color = color_map[node_label]
            color_values.append(color)
    else:
        color_values = ['lightskyblue'] * g.number_of_nodes()
    # print(color_values)
    return color_values


def _gen_color_map(ntypes_count_map):
    fl = len(FAVORITE_COLORS)
    rtn = {}
    # ntypes = defaultdict(int)
    # for g in gs:
    #     for nid, node in g.nodes(data=True):
    #         ntypes[node.get('type')] += 1
    secondary = {}
    for i, (ntype, cnt) in enumerate(
            sorted(ntypes_count_map.items(), key=lambda x: x[1], reverse=True)):
        if ntype is None:
            color = None
            rtn[ntype] = color
        elif i >= fl:
            cmaps = plt.cm.get_cmap('hsv')
            color = cmaps((i - fl) / (len(ntypes_count_map) - fl))
            secondary[ntype] = color
        else:
            color = mcolors.to_rgba(FAVORITE_COLORS[i])[:3]
            rtn[ntype] = color
    if secondary:
        rtn.update(secondary)
    return rtn


def _plot(g1, color_g1, pos_g1, feat_dict_1, g2, color_g2, pos_g2, feat_dict_2,
          dir, fn, need_eps, print_path):
    plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=0, hspace=0)  # set the spacing between axes
    ax = plt.subplot(gs[0])
    ax.axis('off')
    _plot_one_graph(g1, color_g1, pos_g1, feat_dict_1, 900)
    ax = plt.subplot(gs[1])
    ax.axis('off')
    _plot_one_graph(g2, color_g2, pos_g2, feat_dict_2, 900)
    _save_fig(plt, dir, fn, need_eps, print_path)
    plt.close()

    # plt.figure(figsize=(8, 8))
    # plt.tight_layout()
    # plt.axis('off')
    # _plot_one_graph(g1, color_g1, pos_g1, feat_dict_1, 1500)
    # _save_fig(plt, dir, fn + '_g1', print_path)
    # plt.close()
    #
    # plt.figure(figsize=(8, 8))
    # plt.tight_layout()
    # plt.axis('off')
    # _plot_one_graph(g2, color_g2, pos_g2, feat_dict_2, 1500)
    # _save_fig(plt, dir, fn + '_g2', print_path)
    # plt.close()


def _plot_one_graph(g, color, pos, feat_dict, node_size):
    nx.draw_networkx(
        g, node_color=color, pos=pos,
        with_labels=True, labels=feat_dict, node_size=node_size, width=3)


def _save_fig(plt, dir, fn, need_eps=False, print_path=False):
    plt_cnt = 0
    if dir is None or fn is None:
        return plt_cnt
    final_path_without_ext = join(dir, fn)
    exts = ['.png']
    if need_eps:
        exts += '.eps'
    for ext in exts:
        final_path = append_ext_to_filepath(ext, final_path_without_ext)
        create_dir_if_not_exists(dirname(final_path))
        try:
            plt.savefig(final_path, bbox_inches='tight')
        except:
            warn('savefig')
        if print_path:
            print('Saved to {}'.format(final_path))
        plt_cnt += 1
    return plt_cnt


def _sorted_dict(d):
    rtn = OrderedDict()
    for k in sorted(d.keys()):
        rtn[k] = d[k]
    return rtn


if __name__ == '__main__':
    from dataset_config import get_dataset_conf
    from load_data import load_dataset
    from utils import get_temp_path

    dataset_name = 'aids700nef'
    gid1 = 42
    gid2 = 47
    dataset = load_dataset(dataset_name, 'all', 'mcs', 'bfs')
    natts, *_ = get_dataset_conf(dataset_name)
    node_feat_name = natts[0] if len(natts) >= 1 else None
    g1 = dataset.look_up_graph_by_gid(gid1).get_nxgraph()
    g2 = dataset.look_up_graph_by_gid(gid2).get_nxgraph()
    pair = dataset.look_up_pair_by_gids(gid1, gid2)
    mapping = pair.get_y_true_list_dict_view()[0]
    print(mapping)
    dir = get_temp_path()
    fn = '{}_{}_{}'.format(dataset_name, g1.graph['gid'], g2.graph['gid'])
    need_eps = True
    print_path = True
    plot_node_mapping(g1, g2, mapping, node_feat_name, False, dir, fn, need_eps,
                      print_path)
