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
import datetime

import pandas as pd
from nlpia.data.loaders import read_csv

from anomalous import __version__
from anomalous.constants import logging, parse_config
from anomalous.constants import DEFAULT_DB_CSV_FILENAME, DEFAULT_DB_DIR, DEFAULT_CONFIG_FILENAME
from anomalous.utils import stdout_logging, argparse_open_file, argparse_datetime_span, clean_dd_df, update_db
from anomalous.utils import plot_predictions, update_config, clean_dd_all

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
        '-d', '--db', '--database',
        dest="db", required=False,
        help="Path to a directory containing .csv.gz, .pkl, and .cfg files containing historical data, models, and configuration files.",
        type=str,
        metavar="DIR")
    parser.add_argument(
        '--db_cfg',
        dest="db_cfg", required=False,
        help="Path to a .cfg file containing the configuration for chasedown (timespands, hosts, etc).",
        type=str,
        metavar="CFGFILE")
    parser.add_argument(
        '--db_csv',
        dest="db_csv", required=False,
        help="Path to a .csv.gz.",
        type=str,
        metavar="CSVFILE")
    parser.add_argument(
        '-t', '--timespan',
        dest="timespan", required=False,
        help="time span to query datadog for metrics and append to a local 'database' (csv) of historical data, e.g. 'past 24 hours'",
        type=lambda s: argparse_datetime_span(parser, s, allow_none=True),
        metavar="STR")
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
    args.db = args.db or DEFAULT_DB_DIR
    args.db_cfg = args.db_cfg or os.path.join(args.db, DEFAULT_CONFIG_FILENAME)
    cfg = parse_config(path=args.db_cfg, section='chasedown')
    cfg.timespan = argparse_datetime_span(None, getattr(cfg, 'timespan', None), allow_none=True)
    msg = "Config:\n{}".format(cfg.__dict__)
    logger.info(msg)
    cfg = update_config(cfg, args)
    cfg.db_csv = cfg.db_csv or os.path.join(cfg.db, DEFAULT_DB_CSV_FILENAME)
    msg = "cfg.update(args):\n{}".format(cfg.__dict__)
    logger.info(msg)

    # TODO: datadog query is first priority, but can't we do both a query and a file?
    if isinstance(cfg.timespan, tuple):
        start, end = cfg.timespan
    elif cfg.file_or_none is None:
        now = datetime.datetime.now()
        start, end = (now - datetime.timedelta(1), now)

    if cfg.file_or_none is None:
        df = update_db(metric_names=cfg.metrics, start=start, end=end, db=cfg.db_csv, drop=False, save=False)
    else:
        df = clean_dd_df(cfg.file_or_none)

    msg = "(Down)loaded {} series from {} with shape {}".format(
        len(df.columns), cfg.file_or_none, df.shape)
    logger.info(msg)
    logger.debug(df.describe())

    if os.path.isfile(cfg.db_csv):
        db = read_csv(cfg.db_csv)
    else:
        db = pd.DataFrame()
    db = db.append(df)

    db = clean_dd_all(db)
    plot_predictions(db[datetime.datetime.now() - datetime.timedelta(1):].values)


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
