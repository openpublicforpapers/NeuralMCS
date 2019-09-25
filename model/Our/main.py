#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

from config import FLAGS
from data_model import load_train_test_data
from train import train, test
from utils_our import get_model_info_as_str, check_flags, \
    convert_long_time_to_str
from utils import slack_notify, get_ts
from saver import Saver
from eval import Eval
from time import time
from os.path import basename
import traceback


def main():
    if FLAGS.tvt_strategy == 'holdout':
        train_data, test_data = load_train_test_data()
        print('Training...')
        trained_model = train(train_data, saver)
        if FLAGS.save_model:
            saver.save_trained_model(trained_model)
        if FLAGS.debug:
            print('Debugging: Feed train data for testing and eval')
            test_data = train_data
        print('Testing...')
        test(test_data, trained_model, saver)
        eval = Eval(trained_model, train_data, test_data, saver)
        eval.eval_on_test_data()
    elif '-fold' in FLAGS.tvt_strategy:
        # for fold in range(10):
        #     if _train_model(..., saver):
        #         ...
        raise NotImplementedError()
    else:
        assert False
    overall_time = convert_long_time_to_str(time() - t)
    print(overall_time)
    print(saver.get_log_dir())
    print(basename(saver.get_log_dir()))
    saver.save_overall_time(overall_time)
    saver.close()


if __name__ == '__main__':
    t = time()
    print(get_model_info_as_str())
    check_flags()
    saver = Saver()
    try:
        main()
    except:
        traceback.print_exc()
        s = '\n'.join(traceback.format_exc(limit=-1).split('\n')[1:])
        saver.save_exception_msg(traceback.format_exc())
        slack_notify('{}@{}: model {} {} error \n{}'.
                     format(FLAGS.user, FLAGS.hostname, FLAGS.model, get_ts(), s))
    else:
        slack_notify('{}@{}: model {} {} complete'.
                     format(FLAGS.user, FLAGS.hostname, FLAGS.model, get_ts()))
