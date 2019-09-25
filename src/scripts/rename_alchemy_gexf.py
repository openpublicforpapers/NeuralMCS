from glob import glob
from os.path import join, basename
import re
import os
import sys

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

def rename_graphs(dir, base=0):
    for file in sorted_nicely(glob(join(dir, '*.gexf'))):
        gid = int(basename(file).split('.')[0].split('_')[-1])
        gid += int(base)
        os.rename(file, join(dir, '{}.gexf'.format(gid)))

if __name__ == "__main__":
    if 'python' in sys.argv[0]:
        data_path = sys.argv[2]
        base = sys.argv[3]

    else:
        data_path = sys.argv[1]
        base = sys.argv[2]
    rename_graphs(data_path, base)