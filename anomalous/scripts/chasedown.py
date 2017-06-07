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

from anomalous.utils import stdout_logging, argparse_open_file, argparse_datetime_span, clean_df, get_dd_metrics

from anomalous import __version__
from anomalous.constants import logging  # , DATA_PATH

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
        description="(Down)load DataDog json files into a pandas DataFrame (flatfile DB) and interactively flag anomalies")
    parser.add_argument(
        '--version',
        action='version',
        version='anomalous {ver}'.format(ver=__version__))
    parser.add_argument(
        '-l', '--load',
        dest="file_or_none", required=False,  # default=os.path.join(DATA_PATH, 'dd', 'bing_nodes_online', 'day_1.json'),
        help="Path to a DataDog json dump of server monitor time series, e.g. anomalous/data/dd/bing_nodes_online/day_1.json",
        type=lambda s: argparse_open_file(parser, s, mode='r', allow_none=True),
        metavar="FILE")
    parser.add_argument(
        '-u', '--update',
        dest="span", required=False, default='24 hours ago',
        help="Query datadog for past 24 hours of data and append to local 'database' (csv) of historical metric data.",
        type=lambda s: argparse_datetime_span(parser, s, allow_none=True),
        metavar="UPDATE")
    parser.add_argument(
        '-s', '--servers',
        help="Servers (FQDN or hostnames) for servers with Datadog monitor metrics to retrieve.",
        type=str,
        metavar="SERVERS")
    parser.add_argument(
        '-m', '--metrics',
        help="Metric names or filters on metric names.",
        type=str,
        metavar="METRICS")
    parser.add_argument(
        '-v', '--verbose',
        dest="loglevel", default=logging.WARN,
        help="Verbose stdout logging (set loglevel to INFO).",
        action='store_const',
        const=logging.INFO)
    parser.add_argument(
        '-vv', '--very-verbose',
        dest="loglevel",
        help="Very verbose logging (set loglevel to DEBUG).",
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
    msg = "Arguments:\n{}".format(args.__dict__)
    logger.info(msg)
    print(msg)
    if args.file_or_none is None:
        if isinstance(args.span, tuple):
            df = get_dd_metrics(metric_name=args.metrics, servers=args.servers, start=args.span[0], end=args.span[1])
    else:
        df = clean_df(args.file_or_none)
    msg = "Loaded {} series from {} with shape {}:\n{}".format(
        len(df.columns), args.file_or_none, df.shape, df.describe())
    # print(msg)
    logger.info(msg)


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
