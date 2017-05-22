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
import os

from anomalous.utils import stdout_logging, argparse_open_file, clean_df

from anomalous import __version__
from anomalous.constants import logging, DATA_PATH

__author__ = "Hobson Lane"
__copyright__ = "AuthorityLabs"
__license__ = "none"

logger = logging.getLogger(__name__)


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
        "--load",
        dest="file_or_none",
        default=os.path.join(DATA_PATH, 'dd', 'bing_nodes_online', 'day_1.json'),
        required=False,
        help="Path to a DataDog json dump of server monitor time series",
        type=lambda s: argparse_open_file(parser, s, mode='r', allow_none=True),
        metavar="FILE")
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


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str,...]): command line parameter list
    """
    args = parse_args(args)
    stdout_logging(args.loglevel)
    msg = "Loading {}...".format(args.file_or_none)
    logger.debug(msg)
    print(msg)
    df = clean_df(args.file_or_none)
    msg = "Loaded {} series from {} with shape {}:\n{}".format(
        len(df.columns), args.file_or_none, df.shape, df.describe())
    print(msg)
    logger.debug(msg)


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
