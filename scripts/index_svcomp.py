from glob import glob
import argparse
import os
import json
import bz2
import xml.etree.ElementTree as ET
from tqdm import tqdm
import csv


bench_prefix = "../sv-benchmarks/c/"


def _eval_regular(status, filename, prop):
    if 'true' in status:
        if 'true-'+prop in filename:
            return 1
        else:
            return -1
    if 'false' in status:
        if 'valid' in status and prop not in status:
            return 0
        if 'false-'+prop in filename:
            return 1
        else:
            return -1
    return 0


def _eval_memsafety(status, filename, prop):
    if 'true' in status:
        if 'true-valid-memsafety' in filename:
            return 1
        else:
            return -1
    if 'false' in status:
        if prop not in status:
            return 0
        if 'false-'+prop in filename:
            return 1
        else:
            return -1
    return 0


def eval(status, filename, prop):

    if 'valid' not in prop:
        return _eval_regular(status, filename, prop)

    fp = 0
    fn = 0
    tr = 0

    for p in prop.split(" "):
        ev = _eval_memsafety(status, filename, p)
        if ev == -1:
            if 'true' in status:
                fp += 1
            if 'false' in status:
                fn += 1
        elif ev == 1:
            tr += 1

    if fp > 0 or fn > 0:
        return -1
    if tr > 0:
        return 1
    return 0


def get_name(filename, prop):
    sp = filename.split("_")
    name = ""

    for s in sp:
        if s.startswith("true") or s.startswith("false"):
            break
        name = name+"_"+s

    prop = prop.split(" ")[0]

    if 'valid' in prop:
        prop = 'valid-memsafety'

    p = 'true' if 'true-'+prop in filename else 'false'
    return name[1:] + "_" + p + '-' + prop


def parse_xml(file, results):
    with open(file, "rb") as i:
        xmlStr = bz2.decompress(i.read()).decode('utf-8')

    global bench_prefix

    root = ET.fromstring(xmlStr)
    tool_name = root.attrib['benchmarkname']

    for run in root.iter('run'):
        file = run.attrib['name'].replace(bench_prefix, "")
        prop = run.attrib['properties']
        name = get_name(file, prop)

        solve = 0
        solve_time = 0.0

        for c in run:
            if c.attrib['title'] == 'cputime':
                solve_time = float(c.attrib['value'][:-1])
            if c.attrib['title'] == 'status':
                status = c.attrib['value']
                solve = eval(status, name, prop)
                print("%s :: %s ==> %d" % (status, name, solve))

        if file not in results:
            results[file] = {}
        res = results[file]

        prop = prop.split(" ")[0]

        if 'valid' in prop:
            prop = 'valid-memsafety'

        if prop not in res:
            res[prop] = {}
        res = res[prop]
        res['name'] = name
        res[tool_name] = {
            'solve': solve,
            'time': solve_time
        }


def parse_csv(file, results):
    with open(file, "r") as i:
        reader = csv.reader(i, delimiter='\t', quotechar='|')

        global bench_prefix

        for i, row in enumerate(reader):
            if i == 0:
                tool_name = row[1]
            elif i > 2:
                file = row[0].replace(bench_prefix, "")
                prop = "unreach-call"
                name = get_name(file, prop)

                solve = eval(row[1], name, prop)
                solve_time = float(row[2])

                if file not in results:
                    results[file] = {}
                res = results[file]

                if prop not in res:
                    res[prop] = {}
                res = res[prop]
                res['name'] = name
                res[tool_name] = {
                    'solve': solve,
                    'time': solve_time
                }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="directory of XML files created by Benchexec")
    parser.add_argument("out", type=str, help="Json that stores an index for labels")
    parser.add_argument("-c", "--csv", action="store_true", help="Parse CSV files instead of XML.")
    parser.add_argument("-p", "--prefix", type=str)

    args = parser.parse_args()

    if args.prefix:
        bench_prefix = args.prefix

    results = {}

    for f in tqdm(glob(os.path.join(args.dir,
                                    '*.xml.bz2' if not args.csv else '*.csv'))):
        if args.csv:
            parse_csv(f, results)
        else:
            parse_xml(f, results)

    with open(args.out, 'w') as o:
        json.dump(results, o, indent=4)
