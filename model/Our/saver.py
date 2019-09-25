from config import FLAGS
from utils import get_ts, create_dir_if_not_exists, save
from utils_our import get_our_dir, get_model_info_as_str, \
    get_model_info_as_command, extract_config_code
from tensorboardX import SummaryWriter
from collections import OrderedDict
from pprint import pprint
from os.path import join
import torch


class Saver(object):
    def __init__(self):
        model_str = self._get_model_str()
        self.logdir = join(
            get_our_dir(),
            'logs',
            '{}_{}'.format(model_str, get_ts()))
        create_dir_if_not_exists(self.logdir)
        self.writer = SummaryWriter(self.logdir)
        self.model_info_f = self._open('model_info.txt')
        self._log_model_info()
        self._save_conf_code()
        print('Logging to {}'.format(self.logdir))

    def _open(self, f):
        return open(join(self.logdir, f), 'w')

    def close(self):
        self.writer.close()
        self.model_info_f.close()
        if hasattr(self, 'train_log_f'):
            self.train_log_f.close()
        if hasattr(self, 'results_f'):
            self.results_f.close()

    def get_log_dir(self):
        return self.logdir

    def log_model_architecture(self, model):
        self.model_info_f.write('{}\n'.format(model))

    def log_tvt_info(self, s):
        print(s)
        if not hasattr(self, 'train_log_f'):
            self.train_log_f = self._open('train_log.txt')
        self.train_log_f.write('{}\n'.format(s))

    # def log_val_results(self, results, iter):
    #     for metric, num in results.items():
    #         if metric == 'val_loss':
    #             metric = 'total_loss'
    #         summary = tf.Summary(
    #             value=[tf.Summary.Value(tag=metric, simple_value=num)])
    #         self.vw.add_summary(summary, iter)
    #
    # def save_train_val_info(self, train_costs, train_times, ss,
    #                         val_results_dict):
    #     sfn = '{}/train_val_info'.format(self.logdir)
    #     flags = FLAGS.flag_values_dict()
    #     ts = get_ts()
    #     save_as_dict(sfn, train_costs, train_times, val_results_dict, flags, ts)
    #     with open(join(self.logdir, 'train_log.txt'), 'w') as f:
    #         for s in ss:
    #             f.write(s + '\n')

    # def save_test_info(self, sim_dist_mat, time_li, best_iter,
    #                    node_embs_dict, graph_embs_mat, emb_time, atts):
    #     self._save_to_result_file(best_iter, 'best iter')
    #     sfn = '{}/test_info'.format(self.logdir)
    #     # The following function call must be made in one line!
    #     save_as_dict(sfn, sim_dist_mat, time_li, best_iter, node_embs_dict, graph_embs_mat, emb_time, atts)
    #
    # def save_test_result(self, test_results):
    #     self._save_to_result_file(test_results, 'test results')
    #     sfn = '{}/test_result'.format(self.logdir)
    #     # The following function call must be made in one line!
    #     save(sfn, test_results)

    def _save_conf_code(self):
        with open(join(self.logdir, 'config.py'), 'w') as f:
            f.write(extract_config_code())
        p = join(self.get_log_dir(), 'FLAGS')
        save({'FLAGS': FLAGS}, p, print_msg=False)

    def save_trained_model(self, trained_model):
        p = join(self.logdir, 'trained_model.pt')
        torch.save(trained_model.state_dict(), p)
        print('Trained model saved to {}'.format(p))

    def save_eval_result_dict(self, result_dict, label):
        self._save_to_result_file(label)
        self._save_to_result_file(result_dict)

    def save_pairs_with_results(self, test_data, train_data, info):
        p = join(self.get_log_dir(), '{}_pairs'.format(info))
        save({'test_data_pairs':
                  self._shrink_space_pairs(test_data.dataset.pairs),
              # 'train_data_pairs':
              # self._shrink_space_pairs(train_data.dataset.pairs)
              },
             p, print_msg=False)

    def save_ranking_mat(self, true_m, pred_m, info):
        p = join(self.get_log_dir(), '{}_ranking_mats'.format(info))
        save({'true_m': true_m.__dict__, 'pred_m': pred_m.__dict__},
             p, print_msg=False)

    def save_global_eval_result_dict(self, global_result_dict):
        p = join(self.get_log_dir(), 'global_result_dict')
        save(global_result_dict, p, print_msg=False)

    def save_overall_time(self, overall_time):
        self._save_to_result_file(overall_time, 'overall time')

    # def clean_up_saved_models(self, best_iter):
    #     for file in glob('{}/models/*'.format(self.get_log_dir())):
    #         if str(best_iter) not in file:
    #             system('rm -rf {}'.format(file))

    def save_exception_msg(self, msg):
        with self._open('exception.txt') as f:
            f.write('{}\n'.format(msg))

    def _get_model_str(self):
        li = []
        key_flags = [FLAGS.model, FLAGS.dataset]
        for f in key_flags:
            li.append(str(f))
        return '_'.join(li)

    def _log_model_info(self):
        s = get_model_info_as_str()
        c = get_model_info_as_command()
        self.model_info_f.write(s)
        self.model_info_f.write('\n\n')
        self.model_info_f.write(c)
        self.model_info_f.write('\n\n')
        self.writer.add_text('model_info_str', s)
        self.writer.add_text('model_info_command', c)

    def _save_to_result_file(self, obj, name=None):
        if not hasattr(self, 'results_f'):
            self.results_f = self._open('results.txt')
        if type(obj) is dict or type(obj) is OrderedDict:
            # self.f.write('{}:\n'.format(name))
            # for key, value in obj.items():
            #     self.f.write('\t{}: {}\n'.format(key, value))
            pprint(obj, stream=self.results_f)
        elif type(obj) is str:
            self.results_f.write('{}\n'.format(obj))
        else:
            self.results_f.write('{}: {}\n'.format(name, obj))

    def _shrink_space_pairs(self, pairs):
        for _, pair in pairs.items():
            # print(pair.__dict__)
            pair.shrink_space_for_save()
            # pass
            # print(pair.__dict__)
            # exit(-1)
        return pairs
