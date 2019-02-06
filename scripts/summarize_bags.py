import argparse
import json
from glob import glob
import os
import re


def parse_path(path):
    fpath = os.path.basename(path)

    if(len(path) == 0):
        raise ValueError("Path is a directory instead of a file: %s" % path)

    m = re.match("(.*)_([0-9]+)_([0-9]+).json", fpath)

    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    raise ValueError("%s doesn't match the path convention TASKNAME_ITERATION_DEPTH.json" % fpath)


def collect(input_dir):
    result = {}
    for f in glob(os.path.join(input_dir, '*.json')):
        name, it, d = parse_path(f)

        if (it, d) not in result:
            result[(it, d)] = {}

        with open(f, 'r') as i:
            result[(it, d)][name] = json.load(i)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print("Unknown path (%s). EXIT" % args.input_dir)
        exit()

    result = collect(args.input_dir)

    for (i, d), D in result.items():
        path = os.path.join(args.output_dir, "Kernel_%d_%d.json" % (i, d))
        print("Write to %s" % path)
        with open(path, "w") as o:
            json.dump(D, o, indent=4)
