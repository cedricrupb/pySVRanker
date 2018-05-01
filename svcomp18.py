"""
Python module for reading SV-COMP 2015 results into memory.
"""

from lxml import objectify
from enum import Enum, unique
import re
import os
import pandas as pd
from tqdm import tqdm
from .svcomp15 import Status, MissingPropertyTypeException
from .svcomp15 import MissingExpectedStatusException, InvalidDataException


@unique
class PropertyType(Enum):
    unreachability = 1
    memory_safety = 2
    termination = 3
    overflow = 4


_df_cols = ['options', 'status', 'status_msg', 'cputime', 'walltime', 'mem_usage',
            'expected_status', 'property_type']


def select_propety(prop_id):
    if prop_id == 'reach':
        return PropertyType.unreachability
    elif prop_id == 'memory':
        return PropertyType.memory_safety
    elif prop_id == 'termination':
        return PropertyType.termination
    elif prop_id == 'overflow':
        return PropertyType.overflow


def read_category(results_xml_raw_dir_path, category,
                  max_size, witnesscheck=True, prefix='.'):
    """
    Reads a directory of raw xml SVCOMP results into a data frame.

    :param results_xml_raw_dir_path: Path to the raw xml results from SVCOMP
    :return: Pandas data frame
    """

    print('Reading category {} from {}'.format(category, results_xml_raw_dir_path))

    pattern = re.compile(r'\w+\.[0-9-_]+\.(witnesscheck\.[0-9-_]+\.)?results\.sv-comp18\.{0}\.xml'.format(category))
    category_results = {}
    category_witnesschecks = {}
    for file in tqdm(os.listdir(results_xml_raw_dir_path)):
        match = pattern.match(file)
        if match is not None:
            benchmark, df = svcomp_xml_to_dataframe(os.path.join(results_xml_raw_dir_path, file), max_size, prefix)
            if benchmark in category_witnesschecks or benchmark in category_results:
                raise InvalidDataException('Already added data for benchmark {0}'.format(benchmark))
            if 'witnesscheck' not in benchmark:
                category_results[benchmark] = df
            else:
                category_witnesschecks[benchmark] = df
    if witnesscheck:
        _apply_witnesschecks_on_results(category_results, category_witnesschecks)
    return category_results


def svcomp_xml_to_dataframe(xml_path, max_size, prefix):
    """
    Reads raw xml SVCOMP results into a data frame.

    :param xml_path:
    :return:
    """
    with open(xml_path) as f:
        xml = f.read()
    path_prefix = os.path.dirname(xml_path)
    path_prefix = os.path.abspath(path_prefix + prefix)
    root = objectify.fromstring(xml)
    df = pd.DataFrame(columns=_df_cols)
    if hasattr(root, 'run'):
        for source_file in root.run:
            vtask_path = '/'.join([path_prefix, source_file.attrib['name']])

            if os.path.isfile(vtask_path):
                # Ignore too large files
                vtask_size_kb = os.path.getsize(vtask_path) / 1024
                if vtask_size_kb > max_size:
                    print('Too large file detected. Skipping file ' + vtask_path)
                    continue
            else:
                print('Non existing file (%s) detected. Skipping...'
                % vtask_path)
                continue

            r = _columns_to_dict(source_file.column)
            df.loc[vtask_path] = [source_file.attrib['options'] if 'options' in source_file.attrib else '',
                                  _match_status_str(r['status']),
                                  r['status'],
                                  float(r['cputime'][:-1]),
                                  float(r['walltime'][:-1]),
                                  int(r['memUsage']),
                                  _extract_expected_status(vtask_path),
                                  _extract_property_type(vtask_path)]
    return root.attrib['benchmarkname'], df


def _columns_to_dict(columns):
    """
    Simple helper function, which converts column tags to a dictionary.
    :param columns: Collection of column tags
    :return: Dictionary that contains all the information from the columns.
    """
    ret = {}
    for column in columns:
        ret[column.attrib['title']] = column.attrib['value']
    assert ret, 'Could not read columns from sourcefile'
    return ret


def _match_status_str(status_str):
    """
    Maps status strings to their associated meaning
    :param status_str: the status string
    :return: true, false, or unknown
    """
    if re.search(r'true', status_str):
        return Status.true
    if re.search(r'false', status_str):
        return Status.false
    else:
        return Status.unknown


def _extract_expected_status(vtask_path):
    """
    Extracts the expected status from a verification task.

    :param vtask_path: Path to a SVCOMP verification task.
    :return: A tuple containing a verification task's expected result and
             property type if the filename adheres the naming convention.
             TODO What is the exact naming convention?

             Otherwise the result is None.
    """
    match = re.match(r'[-a-zA-Z0-9_\.]+_(true|false)-([-a-zA-Z0-9_\.]+)\.(i|c)',
                     os.path.basename(vtask_path))
    if match is not None:
        return _match_status_str(match.group(1))
    raise MissingExpectedStatusException('Cannot extract expected status from filename / regex failed (wrong naming?)')


def _extract_property_type(vtask_path):
    """
    Extracts the property type associated with a verification task.
    :param vtask_path: path to verification task
    :return: the property type
    """

    # TODO: Elimate this
    if 'ReachSafety' or 'unreach' in vtask_path:
        return PropertyType.unreachability
    if 'MemSafety' or 'valid' in vtask_path:
        return PropertyType.memory_safety
    if 'Overflow' or 'overflow' in vtask_path:
        return PropertyType.overflow
    if 'Termination' or 'termination' in vtask_path:
        return PropertyType.termination

    print(vtask_path)


    unreachability_pattern = re.compile(r'CHECK\([_\s\w\(\)]+,\s*LTL\(\s*G\s*!\s*call\([_\w\s\(\)]+\)\s*\)\s*\)')
    memory_safety_pattern = re.compile(r'CHECK\([_\s\w\(\)]+,\s*LTL\(\s*G\s*valid-\w+\)\s*\)')
    termination_pattern = re.compile(r'CHECK\([_\s\w\(\)]+,\s*LTL\(\s*F\s*end\s*\)\s*\)')
    overflow_pattern = re.compile(r'CHECK\([_\s\w\(\)]+,\s*LTL\(\s*G\s*!\s*overflow\s*\)\s*\)')

    root, ext = os.path.splitext(vtask_path)
    prp = root + '.prp'
    if not os.path.isfile(prp):
        prp = os.path.join(os.path.dirname(vtask_path), 'ALL.prp')
    if not os.path.isfile(prp):
        raise MissingPropertyTypeException('Missing ALL.prp or {filename}.prp')

    with open(prp) as f:
        prp_file_content = f.read()
    if unreachability_pattern.search(prp_file_content) is not None:
        return PropertyType.unreachability
    if memory_safety_pattern.search(prp_file_content) is not None:
        return PropertyType.memory_safety
    if termination_pattern.search(prp_file_content) is not None:
        return PropertyType.termination
    if overflow_pattern.search(prp_file_content) is not None:
        return PropertyType.overflow
    raise MissingPropertyTypeException('Cannot determine property type from prp file')


def score(status, expected_status):
    """
    Evaluation of the result of a verification tool
    :param status: observed status for the analyzed program
    :param expected_status: expected status for the analyzed program
    :return:
    """
    if status is Status.unknown:
        return 0
    if status is Status.false and expected_status is Status.false:
        return 1 # correct
    if status is Status.false and expected_status is Status.true:
        return -6 # false positive
    if status is Status.true and expected_status is Status.true:
        return 2 # correct
    if status is Status.true and expected_status is Status.false:
        return -12 # false negative
    if pd.isnull(status) or pd.isnull(expected_status):
        raise InvalidDataException('Cannot compare status if it is not available (NaN)')
    assert False, 'No score for combination {0} and {1}'.format(status, expected_status)


def compare_results(result_a, result_b):
    """
    Todo
    :param result_a:
    :param result_b:
    :return:
    """
    score_a = score(result_a['status'], result_a['expected_status'])
    score_b = score(result_b['status'], result_b['expected_status'])
    if score_a > score_b:
        return 1
    if score_b > score_a:
        return -1
    if result_a['cputime'] > result_b['cputime']:
        return -1
    if result_b['cputime'] > result_a['cputime']:
        return 1
    return 0


def _apply_witnesschecks_on_results(results, witnesschecks):
    """
    Todo
    :param results:
    :param witnesschecks:
    :return:
    """
    for key in results.keys():
        wcs = [wc for wc in witnesschecks.keys() if wc.startswith('{0}.'.format(key))]
        if len(wcs) > 1:
            raise InvalidDataException('PyPRSVT does not support multiple witnesscheck files.')
        for wc in wcs:
            df = pd.concat({key: results[key], wc: witnesschecks[wc]}, axis=1)
            for row in df.iterrows():
                sourcefile, series = row
                if series[key, 'status'] is Status.false:
                    results[key].set_value(sourcefile, 'status',
                                           _apply_witnesscheck_on_status(series[key, 'status'],
                                                                         series[wc, 'status']))


def _apply_witnesscheck_on_status(status, status_witnesscheck):
    """
    Todo
    :param status:
    :param status_witnesscheck:
    :return:
    """
    if status is not Status.false:
        raise InvalidDataException('Sourcefile with status true or unknown does not provide witnesschecks.')
    # witnesscheck was performed internally (e.g. by CBMC)
    if pd.isnull(status_witnesscheck):
        return Status.false
    # external witnesscheck results
    if status_witnesscheck is Status.false:
        return Status.false
    return Status.unknown
