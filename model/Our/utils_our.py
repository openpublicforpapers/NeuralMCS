from solve_parent_dir import cur_folder
from utils import sorted_nicely, get_ts
from config import FLAGS
from os.path import join
import torch


def check_flags():
    # if FLAGS.node_feat_name:
    #     assert (FLAGS.node_feat_encoder == 'onehot')
    # else:
    #     assert ('constant_' in FLAGS.node_feat_encoder)
    # assert (0 < FLAGS.valid_percentage < 1)
    assert (FLAGS.layer_num >= 1)
    assert (FLAGS.batch_size >= 1)
    # assert (FLAGS.num_epochs >= 0)
    # assert (FLAGS.iters_val_start >= 1)
    # assert (FLAGS.iters_val_every >= 1)
    d = vars(FLAGS)
    ln = d['layer_num']
    ls = [False] * ln
    for k in d.keys():
        if 'layer_' in k and 'gc' not in k and 'branch' not in k and 'id' not in k:
            lt = k.split('_')[1]
            if lt != 'num':
                i = int(lt) - 1
                if not (0 <= i < len(ls)):
                    raise RuntimeError('Wrong spec {}'.format(k))
                ls[i] = True
    for i, x in enumerate(ls):
        if not x:
            raise RuntimeError('layer {} not specified'.format(i + 1))
    if 'cuda' in FLAGS.device:
        gpu_id = int(FLAGS.device.split(':')[1])
        gpu_count = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError('Wrong GPU ID {}; {} available GPUs'.
                             format(FLAGS.device, gpu_count))
    # TODO: finish.


def get_flag(k, check=False):
    if hasattr(FLAGS, k):
        return getattr(FLAGS, k)
    else:
        if check:
            raise RuntimeError('Need flag {} which does not exist'.format(k))
        return None


def get_flags_with_prefix_as_list(prefix):
    rtn = []
    d = vars(FLAGS)
    i_check = 1  # one-based
    for k in sorted_nicely(d.keys()):
        v = d[k]
        sp = k.split(prefix)
        if len(sp) == 2 and sp[0] == '' and sp[1].startswith('_'):
            id = int(sp[1][1:])
            if i_check != id:
                raise ValueError('Wrong flag format {}={} '
                                 '(should start from _1'.format(k, v))
            rtn.append(v)
            i_check += 1
    return rtn


def get_branch_names():
    bnames = get_flag('branch_names')
    if bnames:
        rtn = bnames.split(',')
        if len(rtn) == 0:
            raise ValueError('Wrong number of branches: {}'.format(bnames))
        return rtn
    else:
        assert bnames is None
        return None


def extract_config_code():
    with open(join(get_our_dir(), 'config.py')) as f:
        return f.read()


def convert_long_time_to_str(sec):
    def _give_s(num):
        return '' if num == 1 else 's'

    day = sec // (24 * 3600)
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    minutes = sec // 60
    sec %= 60
    seconds = sec
    return '{} day{} {} hour{} {} min{} {:.1f} sec{}'.format(
        int(day), _give_s(int(day)), int(hour), _give_s(int(hour)),
        int(minutes), _give_s(int(minutes)), seconds, _give_s(seconds))


def get_our_dir():
    return cur_folder


def get_model_info_as_str():
    rtn = []
    d = vars(FLAGS)
    for k in sorted_nicely(d.keys()):
        v = d[k]
        s = '{0:26} : {1}'.format(k, v)
        rtn.append(s)
    rtn.append('{0:26} : {1}'.format('ts', get_ts()))
    return '\n'.join(rtn)


def get_model_info_as_command():
    rtn = []
    d = vars(FLAGS)
    for k in sorted_nicely(d.keys()):
        v = d[k]
        s = '--{}={}'.format(k, v)
        rtn.append(s)
    return 'python {} {}'.format(join(get_our_dir(), 'main.py'), '  '.join(rtn))


def debug_tensor(tensor):
    xxx = tensor.detach().cpu().numpy()
    return


TDMNN = None


def get_train_data_max_num_nodes(train_data):
    global TDMNN
    if TDMNN is None:
        TDMNN = train_data.dataset.stats['#Nodes']['Max']
    return TDMNN


def pad_extra_rows(g1x, g2x, padding_value=0):  # g1x and g2x are 2D tensors
    max_dim = max(g1x.shape[0], g2x.shape[0])

    x1_pad = torch.nn.functional.pad(g1x, (0, 0, 0, (max_dim - g1x.shape[0])),
                                     mode='constant',
                                     value=padding_value)
    x2_pad = torch.nn.functional.pad(g2x, (0, 0, 0, (max_dim - g2x.shape[0])),
                                     mode='constant',
                                     value=padding_value)

    return x1_pad, x2_pad
