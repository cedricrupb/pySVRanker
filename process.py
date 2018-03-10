import json
import argparse
from pyTasks.utils import containerHash
from tqdm import tqdm


def indexL(i):
    out = {}
    for k, v in tqdm(i['categories'].items()):
        out[k] = {}
        out[k]['large'] = str(containerHash(v, large=True))
        out[k]['normal'] = str(containerHash(v))

    with open('./index.json', 'w') as o:
        json.dump(out, o, indent=4)


def main():
    """Main method of the executor."""
    parser =\
        argparse.ArgumentParser(description='Create an execution wrapper '
                                            'for config files')
    parser.add_argument('-i', '--index', type=str, required=True)

    args = parser.parse_args()

    if args.index:
        with open(args.index, 'r') as i:
            ind = json.load(i)

        indexL(ind)


if __name__ == '__main__':
    main()
