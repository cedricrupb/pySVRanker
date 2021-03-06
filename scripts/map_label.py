import json
import argparse
import os
from glob import glob
import copy
from tqdm import tqdm


def pathId(path):
    base = os.path.basename(path)
    path = os.path.dirname(path)
    base = os.path.split(path)[1] + '_'+base

    return base.replace(".", "_")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=str, help="File that is generated by index_svcomp")
    parser.add_argument("input_dir", type=str, help="directory of kernels")
    parser.add_argument("output_dir", type=str, help="directory for mapped kernels")

    args = parser.parse_args()

    with open(args.index, "r") as i:
        index = json.load(i)

    nIndex = {}
    for k, V in index.items():
        nIndex[pathId(k)] = V

    del index

    for f in tqdm(glob(args.input_dir+"/*.json")):
        nBag = {}
        with open(f, "r") as i:
            Bag = json.load(i)

            for k, V in Bag.items():
                if k in nIndex:
                    labels = nIndex[k]

                    for type, label in labels.items():
                        label = copy.deepcopy(label)
                        name = label["name"]
                        del label["name"]

                        kernel = copy.deepcopy(V['kernel_bag'])
                        kernel[type] = 1

                        nBag[pathId(name)] = {
                            'file': name,
                            'kernel_bag': kernel,
                            'label': label
                        }
                else:
                    print("Not found: %s" % k)

        path = os.path.join(args.output_dir, os.path.basename(f))
        print("Write to %s" % path)

        with open(path, "w") as o:
            json.dump(nBag, o, indent=4)
