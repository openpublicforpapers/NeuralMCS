from load_classic_data import load_debug_data, load_classic_data
from load_old_data import load_old_data

def get_dataset_conf(name):
    if 'debug' in name:
        natts = ['type']
        eatts = []
        tvt_options = ['all']
        align_metric_options = ['random']
        loader = load_debug_data
        dataset_type = 'OurDataset'
    elif name in ['aids700nef', 'linux', 'imdbmulti', 'redditmulti10k']:
        if name in ['aids700nef']:
            natts = ['type']
        else:
            natts = []
        eatts = []
        tvt_options = ['all']
        align_metric_options = ['mcs']
        loader = load_classic_data
        dataset_type = 'OurDataset'
    else:
        raise ValueError('Unknown dataset {}'.format(name))
    check_tvt_align_lists(tvt_options, align_metric_options)
    return natts, eatts, tvt_options, align_metric_options, loader, dataset_type


def check_tvt_align_lists(tvt_options, align_metric_options):
    for tvt in tvt_options:
        check_tvt(tvt)
    for align_metric in align_metric_options:
        check_align(align_metric)


def check_tvt(tvt):
    if tvt not in ['train', 'val', 'test', 'all']:
        raise ValueError('Unknown tvt specifier {}'.format(tvt))


def check_align(align_metric):
    if align_metric not in ['ged', 'mcs', 'vsm', 'random']:
        raise ValueError('Unknown graph alignment metric {}'.
                         format(align_metric))

