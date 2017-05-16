#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
`python setup.py install` install the command `chasedown` inside your current environment.

For help/docs type `chasedown -h` in your console/shell
"""
from __future__ import print_function, unicode_literals, division, absolute_import
from future import standard_library
standard_library.install_aliases()  # noqa
from builtins import *  # noqa

import argparse
import sys
import logging
import datetime
import json

import pandas as pd

from python3 import __version__

__author__ = "Hobson Lane"
__copyright__ = "AuthorityLabs"
__license__ = "none"

logger = logging.getLogger(__name__)


def clean_series(series):
    """Convert a DataDog "series" dictionary to a Pandas Series"""
    t = pd.Series(pd.np.array(series['pointlist']).T[0],
                  name='datetime')
    t = t.apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.))
    ts = pd.Series(pd.np.array(series['pointlist']).T[1],
                   index=t,
                   name=series['display_name'])
    return ts


def clean_df(filepath):
    """Load a DataDog json file and return a Pandas DataFrame of all the time series in the json list of dicts.

    Args:
      filepath (str): path to json file from DataDog dump

    Returns:
      pd.Series: Pandas Series with the timestamp as the index and a series name composed from the file path
    """
    with open(filepath, 'r') as f:
        js = json.load(f)
        df = pd.DataFrame
        for series in js['series']:
            ts = clean_series(series)
            df[ts.name] = ts
    return df


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Load DataDog json files into a pandas DataFrame")
    parser.add_argument(
        '--version',
        action='version',
        version='anomalous {ver}'.format(ver=__version__))
    parser.add_argument(
        dest="filepath",
        help="n-th Fibonacci number",
        type=int,
        metavar="INT")
    parser.add_argument(
        '-v',
        '--verbose',
        dest="loglevel",
        help="set loglevel to INFO",
        action='store_const',
        const=logging.INFO)
    parser.add_argument(
        '-vv',
        '--very-verbose',
        dest="loglevel",
        help="set loglevel to DEBUG",
        action='store_const',
        const=logging.DEBUG)
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str,...]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    msg = "Loading {}...".format(args.filepath)
    logger.debug(msg)
    print(msg)
    df = clean_df(args.filepath)
    msg = "Loaded {} series from {} with shape {}.".format(
        len(df.columns), args.filepath, df.shape)
    print(msg)
    logger.debug(msg)


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
