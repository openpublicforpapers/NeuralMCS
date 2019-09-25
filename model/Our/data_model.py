from load_data import load_dataset
from node_feat import encode_node_features
from config import FLAGS
from torch.utils.data import Dataset as TorchDataset
import torch
from utils_our import get_flags_with_prefix_as_list
from utils import get_save_path, save, load
from os.path import join
from warnings import warn


class OurModelData(TorchDataset):
    """Stores a list of graph id pairs with known pairwise results."""

    def __init__(self, dataset, num_node_feat):
        self.dataset, self.num_node_feat = dataset, num_node_feat
        gid_pairs = list(self.dataset.pairs.keys())
        self.gid1gid2_list = torch.tensor(
            sorted(gid_pairs),
            device=FLAGS.device)  # takes a while to move to GPU

    def __len__(self):
        return len(self.gid1gid2_list)

    def __getitem__(self, idx):
        return self.gid1gid2_list[idx]

    def get_pairs_as_list(self):
        return [self.dataset.look_up_pair_by_gids(gid1.item(), gid2.item())
                for (gid1, gid2) in self.gid1gid2_list]

    def truncate_large_graphs(self):
        gid_pairs = list(self.dataset.pairs.keys())
        if FLAGS.filter_large_size < 1:
            raise ValueError('Cannot filter graphs of size {} < 1'.format(
                FLAGS.filter_large_size))
        rtn = []
        num_truncaed = 0
        for (gid1, gid2) in gid_pairs:
            g1 = self.dataset.look_up_graph_by_gid(gid1)
            g2 = self.dataset.look_up_graph_by_gid(gid2)
            if g1.get_nxgraph().number_of_nodes() <= FLAGS.filter_large_size and \
                    g2.get_nxgraph().number_of_nodes() <= FLAGS.filter_large_size:
                rtn.append((gid1, gid2))
            else:
                num_truncaed += 1
        warn('{} graph pairs truncated; {} left'.format(num_truncaed, len(rtn)))
        self.gid1gid2_list = torch.tensor(
            sorted(rtn),
            device=FLAGS.device)  # takes a while to move to GPU

    def select_specific_for_debugging(self):
        gid_pairs = list(self.dataset.pairs.keys())
        gids_selected = FLAGS.select_node_pair.split('_')
        assert(len(gids_selected) == 2)
        gid1_selected, gid2_selected = int(gids_selected[0]), int(gids_selected[1])
        rtn = []
        num_truncaed = 0
        for (gid1, gid2) in gid_pairs:
            g1 = self.dataset.look_up_graph_by_gid(gid1).get_nxgraph()
            g2 = self.dataset.look_up_graph_by_gid(gid2).get_nxgraph()
            if g1.graph['gid'] == gid1_selected and g2.graph['gid'] == gid2_selected:
                rtn.append((gid1, gid2))
            else:
                num_truncaed += 1
        warn('{} graph pairs truncated; {} left'.format(num_truncaed, len(rtn)))
        FLAGS.select_node_pair = None # for test
        self.gid1gid2_list = torch.tensor(
            sorted(rtn),
            device=FLAGS.device)  # takes a while to move to GPU

def load_train_test_data():
    # tvt = 'train'
    dir = join(get_save_path(), 'OurModelData')
    sfn = '{}_train_test_{}_{}_{}'.format(
        FLAGS.dataset, FLAGS.align_metric, FLAGS.node_ordering,
        '_'.join(get_flags_with_prefix_as_list('node_fe')))
    '''
    sfn = '{}_train_test_{}_{}_{}{}{}'.format(
        FLAGS.dataset, FLAGS.align_metric, FLAGS.node_ordering,
        '_'.join(get_flags_with_prefix_as_list('node_fe')),
        _none_empty_else_underscore(FLAGS.filter_large_size),
        _none_empty_else_underscore(FLAGS.select_node_pair))
    '''
    tp = join(dir, sfn)
    rtn = load(tp)
    if rtn:
        train_data, test_data = rtn['train_data'], rtn['test_data']
    else:
        train_data, test_data = _load_train_test_data_helper()
        save({'train_data': train_data, 'test_data': test_data}, tp)
    if FLAGS.validation:
        all_spare_ratio = 1 - FLAGS.throw_away
        train_val_ratio = 0.6 * all_spare_ratio
        dataset = train_data.dataset
        dataset.tvt = 'all'
        if all_spare_ratio != 1:
            dataset_train, dataset_test, _ = dataset.tvt_split(
                [train_val_ratio, all_spare_ratio], ['train', 'validation', 'spare'])
        else:
            dataset_train, dataset_test = dataset.tvt_split(
                [train_val_ratio], ['train', 'validation'])
        assert train_data.num_node_feat == test_data.num_node_feat
        train_data = OurModelData(dataset_train, train_data.num_node_feat)
        test_data = OurModelData(dataset_test, test_data.num_node_feat)

    if FLAGS.filter_large_size is not None:
        print('truncating graphs...')
        train_data.truncate_large_graphs()
        test_data.truncate_large_graphs()

    if FLAGS.select_node_pair is not None:
        print('selecting node pair...')
        train_data.select_specific_for_debugging()
        test_data.select_specific_for_debugging()

    train_data.dataset.print_stats()
    test_data.dataset.print_stats()

    dir = join(get_save_path(), 'anchor_data')
    sfn = '{}_{}_{}_{}'.format(
        FLAGS.dataset, FLAGS.align_metric, FLAGS.node_ordering,
        '_'.join(get_flags_with_prefix_as_list('node_fe')))
    tp = join(dir, sfn)
    rtn = load(tp)
    # if rtn:
    #     train_anchor, test_anchor = rtn['train_anchor'], rtn['test_anchor']
    #     train_data.dataset.generate_anchors(train_anchor)
    #     test_data.dataset.generate_anchors(test_anchor)
    # else:
    #     train_anchor = train_data.dataset.generate_anchors(None)
    #     test_anchor = test_data.dataset.generate_anchors(None)
    #     save({'train_anchor': train_anchor, 'test_anchor': test_anchor}, tp)
    #
    # # load to device
    # def load_to_device(dataset, device = FLAGS.device):
    #     for i, g in enumerate(dataset.dataset.gs):
    #         dataset.dataset.gs[i].nxgraph.graph['dists_max'] = g.nxgraph.graph['dists_max'].to(device)
    #         dataset.dataset.gs[i].nxgraph.graph['dists_argmax'] = g.nxgraph.graph['dists_argmax'].to(
    #             device)
    # load_to_device(train_data)
    # load_to_device(test_data)

    return train_data, test_data


def _none_empty_else_underscore(v):
    if v is None:
        return ''
    return '_{}'.format(v)


def _load_train_test_data_helper():
    if FLAGS.tvt_options == 'all':
        dataset = load_dataset(FLAGS.dataset, 'all', FLAGS.align_metric,
                               FLAGS.node_ordering)
        dataset.print_stats()
        # Node feature encoding must be done at the entire dataset level.
        print('Encoding node features')
        dataset, num_node_feat = encode_node_features(dataset=dataset)
        print('Splitting dataset into train test')
        dataset_train, dataset_test = dataset.tvt_split(
            [FLAGS.train_test_ratio], ['train', 'test'])
    elif FLAGS.tvt_options == 'train,test':
        dataset_test = load_dataset(FLAGS.dataset, 'test', FLAGS.align_metric,
                                    FLAGS.node_ordering)
        dataset_train = load_dataset(FLAGS.dataset, 'train', FLAGS.align_metric,
                                     FLAGS.node_ordering)
        dataset_train, num_node_feat_train = \
            encode_node_features(dataset=dataset_train)
        dataset_test, num_node_feat_test = \
            encode_node_features(dataset=dataset_test)
        if num_node_feat_train != num_node_feat_test:
            raise ValueError('num_node_feat_train != num_node_feat_test '
                             '{] != {}'.
                             format(num_node_feat_train, num_node_feat_test))
        num_node_feat = num_node_feat_train
    else:
        print(FLAGS.tvt_options)
        raise NotImplementedError()
    dataset_train.print_stats()
    dataset_test.print_stats()
    train_data = OurModelData(dataset_train, num_node_feat)
    test_data = OurModelData(dataset_test, num_node_feat)
    return train_data, test_data


if __name__ == '__main__':
    from torch.utils.data import DataLoader, random_split
    from torch.utils.data.sampler import SubsetRandomSampler
    from batch import BatchData
    import random

    # print(len(load_dataset(FLAGS.dataset).gs))
    data = OurModelData()
    print(len(data))
    # print('model_data.num_features', data.num_features)
    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(dataset_size * 0.2)
    random.Random(123).shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    loader = DataLoader(data, batch_size=3, shuffle=True)
    print(len(loader.dataset))
    for i, batch_gids in enumerate(loader):
        print(i, batch_gids)
        batch_data = BatchData(batch_gids, data.dataset)
        print(batch_data)
        # print(i, batch_data, batch_data.num_graphs, len(loader.dataset))
        # print(batch_data.sp)
