import networkx as nx
from os.path import dirname, abspath, exists, join, isfile, expanduser
from os import makedirs, system, environ
from socket import gethostname
from collections import OrderedDict
import klepto
import subprocess
from threading import Timer
from time import time
import datetime, pytz
import re
import requests
import pickle
import signal


def check_nx_version():
    nxvg = '2.2'
    nxva = nx.__version__
    if nxvg != nxva:
        raise RuntimeError(
            'Wrong networkx version! Need {} instead of {}'.format(nxvg, nxva))


# Always check the version first.
check_nx_version()


def get_root_path():
    return dirname(dirname(abspath(__file__)))


def get_data_path():
    return join(get_root_path(), 'data')


def get_save_path():
    return join(get_root_path(), 'save')


def get_src_path():
    return join(get_root_path(), 'src')


def get_model_path():
    return join(get_root_path(), 'model')


def get_result_path():
    return join(get_root_path(), 'result')


def get_temp_path():
    return join(get_root_path(), 'temp')


def create_dir_if_not_exists(dir):
    if not exists(dir):
        makedirs(dir)


def save(obj, filepath, print_msg=True):
    if type(obj) is not dict and type(obj) is not OrderedDict:
        raise ValueError('Can only save a dict or OrderedDict'
                         ' NOT {}'.format(type(obj)))
    fp = proc_filepath(filepath)
    create_dir_if_not_exists(dirname(filepath))
    save_klepto(obj, fp, print_msg)


def load(filepath, print_msg=True):
    fp = proc_filepath(filepath)
    if isfile(fp):
        return load_klepto(fp, print_msg)
    elif print_msg:
        print('Trying to load but no file {}'.format(fp))


def save_klepto(dic, filepath, print_msg):
    if print_msg:
        print('Saving to {}'.format(filepath))
    klepto.archives.file_archive(filepath, dict=dic).dump()


def load_klepto(filepath, print_msg):
    rtn = klepto.archives.file_archive(filepath)
    rtn.load()
    if print_msg:
        print('Loaded from {}'.format(filepath))
    return rtn


def load_pickle(filepath, print_msg=True):
    fp = proc_filepath(filepath, '.pickle')
    if isfile(fp):
        with open(fp, 'rb') as handle:
            pickle_data = pickle.load(handle)
            return pickle_data
    elif print_msg:
        print('No file {}'.format(fp))


def proc_filepath(filepath, ext='.klepto'):
    if type(filepath) is not str:
        raise RuntimeError('Did you pass a file path to this function?')
    return append_ext_to_filepath(ext, filepath)


def append_ext_to_filepath(ext, fp):
    if not fp.endswith(ext):
        fp += ext
    return fp


def sorted_nicely(l, reverse=False):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        if type(s) is not str:
            raise ValueError('{} must be a string in l: {}'.format(s, l))
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    rtn = sorted(l, key=alphanum_key)
    if reverse:
        rtn = reversed(rtn)
    return rtn


def exec_cmd(cmd, timeout=None, exec_print=True):
    if not timeout:
        if exec_print:
            print(cmd)
        else:
            cmd += ' > /dev/null'
        system(cmd)
        return True  # finished
    else:
        def kill_proc(proc, timeout_dict):
            timeout_dict["value"] = True
            proc.kill()

        def run(cmd, timeout_sec):
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            timeout_dict = {"value": False}
            timer = Timer(timeout_sec, kill_proc, [proc, timeout_dict])
            timer.start()
            stdout, stderr = proc.communicate()
            timer.cancel()
            return proc.returncode, stdout.decode("utf-8"), \
                   stderr.decode("utf-8"), timeout_dict["value"]

        if exec_print:
            print('Timed cmd {} sec(s) {}'.format(timeout, cmd))
        _, _, _, timeout_happened = run(cmd, timeout)
        if exec_print:
            print('timeout_happened?', timeout_happened)
        return not timeout_happened


tstamp = None


def get_ts():
    global tstamp
    if not tstamp:
        tstamp = get_current_ts()
    return tstamp


def get_current_ts(zone='US/Pacific'):
    return datetime.datetime.now(pytz.timezone(zone)).strftime(
        '%Y-%m-%dT%H-%M-%S.%f')


class timeout:
    """
    https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
    """

    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def get_user():
    try:
        home_user = expanduser("~").split('/')[-1]
    except:
        home_user = 'user'
    return home_user


def get_host():
    host = environ.get('HOSTNAME')
    if host is not None:
        return host
    return gethostname()


# posts to slack channel
def slack_notify(message):
    # redacted
    return


def assert_valid_nid(nid, g):
    assert type(nid) is int and (0 <= nid < g.number_of_nodes())


def assert_0_based_nids(g):
    for i, (n, ndata) in enumerate(sorted(g.nodes(data=True))):
        assert_valid_nid(n, g)
        assert i == n  # 0-based consecutive node ids


def format_str_list(sl):
    assert type(sl) is list
    if not sl:
        return 'None'
    else:
        return ','.join(sl)


class C(object):  # counter
    def __init__(self):
        self.count = 0

    def c(self):  # count and increment itself
        self.count += 1
        return self.count

    def t(self):  # total
        return self.count


class Timer(object):
    def __init__(self):
        self.t = time()

    def time_msec_and_clear(self):
        # from time import sleep
        # sleep(2.5)
        now = time()
        duration = now - self.t
        self.t = now
        # print(duration)
        rtn = format_seconds(duration)
        # print('@@@', rtn, type(rtn), '@@@')
        return rtn


def format_seconds(seconds):
    """
    https://stackoverflow.com/questions/538666/python-format-timedelta-to-string
    """
    periods = [
        ('year', 60 * 60 * 24 * 365),
        ('month', 60 * 60 * 24 * 30),
        ('day', 60 * 60 * 24),
        ('hour', 60 * 60),
        ('min', 60),
        ('sec', 1)
    ]

    if seconds <= 1:
        return '{:.2f} secs'.format(seconds)

    strings = []
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            if period_name == 'sec':
                period_value = seconds
                has_s = 's'
            else:
                period_value, seconds = divmod(seconds, period_seconds)
                has_s = 's' if period_value > 1 else ''
            strings.append('{:.2f} {}{}'.format(period_value, period_name, has_s))

    return ', '.join(strings)
