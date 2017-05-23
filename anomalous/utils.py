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

import datetime
import json
import os
import sys

import pandas as pd
from .constants import logging, DATA_PATH

from pugnlp.futil import find_files
from sklearn.preprocessing import Imputer, MinMaxScaler

DEFAULT_JSON_PATH = os.path.join(DATA_PATH, 'dd', 'bing_nodes_online', 'day_1.json')

__author__ = "Hobson Lane"
__copyright__ = "AuthorityLabs"
__license__ = "none"

logger = logging.getLogger(__name__)


def stdout_logging(loglevel=logging.INFO):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(lineno)d: %(message)s"

    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def argparse_open_file(parser, s, mode='r', allow_none=True):
    if allow_none and s is None:
        return None
    if not os.path.isfile(s):
        parser.error("The file {} does not exist!".format(s))
        return False
    else:
        return open(s, mode=mode)  # return an open file handle


def read_series(file_or_path=None, i=0):
    """Convert a DataDog "series" json file into a Pandas Series"""
    file_or_path = file_or_path or DEFAULT_JSON_PATH
    with (open(file_or_path, 'r') if isinstance(file_or_path, (str, bytes)) else file_or_path) as f:
        js = json.load(f)
        ts = pd.Series(js['series'][i])
    return ts


def clean_series(series):
    """Convert a DataDog "series" dictionary to a Pandas Series"""
    t = pd.Series(pd.np.array(series['pointlist']).T[0],
                  name='datetime')
    t = t.apply(lambda x: datetime.datetime.fromtimestamp(x / 1000.))
    name = series['display_name'].strip().strip('()-=+!._$%#@*[]{}').lower()[:48]
    ts = pd.Series(pd.np.array(series['pointlist']).T[1],
                   index=t.values,
                   name=name)
    return ts


def clean_df(file_or_path=None):
    """Load a DataDog json file and return a Pandas DataFrame of all the time series in the json list of dicts.

    Args:
      filepath (str): path to json file from DataDog dump

    Returns:
      pd.Series: Pandas Series with the timestamp as the index and a series name composed from the file path
    """
    file_or_path = file_or_path or os.path.join(DATA_PATH, 'dd', 'bing_nodes_online', 'day_1.json')
    with (open(file_or_path, 'r') if isinstance(file_or_path, (str, bytes)) else file_or_path) as f:
        js = json.load(f)
        df = pd.DataFrame()
        for series in js['series']:
            ts = clean_series(series)
            df[ts.name] = ts
    return df


def load_all(dirpath=os.path.join(DATA_PATH, 'dd')):
    files = find_files(dirpath, ext='json')
    df = pd.DataFrame()
    for f in files:
        df = df.append(clean_df(f['path']))
    return df


def align_samples(df):
    """Undersample, interpolate, or impute values to fill in NaNs"""
    scaler = MinMaxScaler()
    
    # df = df.interpolate(limit=100000000000, limit_direction='both', axis=0)
    # df = df.resample('1s').agg('mean')
    # df = df.dropna()
    # df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    # imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    # dfimp = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    # dfscal = pd.DataFrame(scaler.fit_transform(dfimp), columns=df.columns, index=df.index)

    df = load_all().astype(float)
    df.sort_index()
    df = df.resample('5min').agg('mean')
    df_scaled = pd.DataFrame(index=df.index)
    for col in df.columns:
        s = df[col].dropna()
        index = s.index
        s = s.values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler = scaler.fit(s)
        df_scaled[col+'x{:6g}'.format(scaler.scale_[0])] = pd.Series(scaler.transform(s)[:,0], index=index)
    return df, df_scaled


def is_anomalous(df):
    # bing crawler_nodes > 10
    # google error rate > 2%
    # papi > 110k
    # google reques > 2k
    # redis connections > 10.5k

    # ex_workers.crawler_nodes.bing                           16.000000
    # papi.queue.web_insight                              155142.000000
    # proper.redis.requeue.standard.google                 19556.767578
    # redis.net.clients                                    11330.875000
    # workers.us.google.status.901 + workers.us.google        41.577825

    d = {
        'ex_workers.crawler_nodes.bing': 10,
        'papi.queue.web_insight': 110000.0,
        'proper.redis.requeue.standard.google': 2000.0,
        'redis.net.clients': 10500.0,
        'workers.us.google.status.901 + workers.us.google': 2.0
        }

    ans = pd.np.zeros(len(df))
    for k, v in d.items():
        ans |= df[k] > v
    return ans