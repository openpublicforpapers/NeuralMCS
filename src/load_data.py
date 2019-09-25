from dataset_config import get_dataset_conf, check_tvt, check_align
from dataset import OurDataset, OurOldDataset
from utils import get_save_path, load, save
from os.path import join


def load_dataset(name, tvt, align_metric, node_ordering):
    name_list = [name]
    if not name or type(name) is not str:
        raise ValueError('name must be a non-empty string')
    check_tvt(tvt)
    name_list.append(tvt)
    check_align(align_metric)
    name_list.append(align_metric)
    if node_ordering is None:
        node_ordering = 'noordering'
    elif node_ordering == 'bfs':
        pass
    else:
        raise ValueError('Unknown node ordering {}'.format(node_ordering))
    name_list.append(node_ordering)
    full_name = '_'.join(name_list)
    p = join(get_save_path(), 'dataset', full_name)
    ld = load(p)
    '''
    ######### this is solely for running locally lol #########
    ld['pairs'] = {(1022,1023):ld['pairs'][(1022,1023)],\
                   (1036,1037):ld['pairs'][(1036,1037)], \
                   (104,105):ld['pairs'][(104,105)],\
                   (1042,1043):ld['pairs'][(1042,1043)],\
                   (1048,1049):ld['pairs'][(1048,1049)],\
                   }
    '''
    if ld:
        _, _, _, _, _, dataset_type = get_dataset_conf(name)
        if dataset_type == 'OurDataset':
            rtn = OurDataset(None, None, None, None, None, None, None, None, ld)
        elif dataset_type == 'OurOldDataset':
            rtn = OurOldDataset(None, None, None, None, None, None, None, None,
                          None, None, ld)
        else:
            raise NotImplementedError()
    else:
        rtn = _load_dataset_helper(name, tvt, align_metric, node_ordering)
        save(rtn.__dict__, p)
    if rtn.num_graphs() == 0:
        raise ValueError('{} has 0 graphs'.format(name))
    return rtn


def _load_dataset_helper(name, tvt, align_metric, node_ordering):
    natts, eatts, tvt_options, align_metric_options, loader, _ = \
        get_dataset_conf(name)
    if tvt not in tvt_options:
        raise ValueError('Dataset {} only allows tvt options '
                         '{} but requesting {}'.
                         format(name, tvt_options, tvt))
    if align_metric not in align_metric_options:
        raise ValueError('Dataset {} only allows alignment metrics '
                         '{} but requesting {}'.
                         format(name, align_metric_options, align_metric))
    assert loader
    return loader(name, natts, eatts, tvt, align_metric, node_ordering)


if __name__ == '__main__':
    name = 'imdbmulti'
    dataset = load_dataset(name, 'all', 'mcs', 'bfs')
    # print(dataset)
    # print(dataset.gs)
    dataset.print_stats()
    # pair = dataset.look_up_pair_by_gids(165, 20679)
    # print(pair)
    # g1 = dataset.look_up_graph_by_gid(165)
    # g2 = dataset.look_up_graph_by_gid(20679)
    print()
