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

from pugnlp.futil import find_files
from sklearn.preprocessing import MinMaxScaler

import plotly
from plotly import offline
import cufflinks

from .constants import logging, DATA_PATH


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


def clean_all(df=None, fillna_method='ffill', dropna=True, consolidate_index=False):
    """Undersample, interpolate, or impute values to fill in NaNs"""
    df = os.path.join(DATA_PATH, 'dd') if df is None else df
    df = load_all(df).astype(float) if isinstance(df, str) else pd.DataFrame(df)

    df.sort_index(inplace=True)
    df = df.reindex()
    if consolidate_index:
        df = df[~df.index.duplicated(keep='last')]
        df = df.reindex()
    df.fillna(method=fillna_method, inplace=True, axis=0)
    # ^ this leaves a few NaNs at the beginning, start them at 0:
    if dropna is True:
        df.dropna(inplace=True)
    else:
        df.fillna(value=0, inplace=True)
    return df


def scale_all(df=None, fillna_method='ffill', dropna=False):
    df = os.path.join(DATA_PATH, 'dd') if df is None else df
    df = clean_all(df, fillna_method=fillna_method, dropna=dropna).astype(float) if isinstance(df, str) else pd.DataFrame(df)

    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), index=df.index)
    df.columns = ['{}x{:6g}'.format(c, s) for s, c in zip(scaler.scale_, df.columns)]
    return df


def is_anomalous(df, thresholds=None):
    """ Compose DataFrame indicating which signal exceed predetermined thresholds

    Example Thresholds:
    
    - bing crawler_nodes > 10
    - google error rate > 2%
    - papi > 110k
    - google reques > 2k
    - redis connections > 10.5k

    # >>> df.max()
    # ex_workers.crawler_nodes.bing                           16.000000
    # papi.queue.web_insight                              155142.000000
    # proper.redis.requeue.standard.google                 19556.767578
    # redis.net.clients                                    11330.875000
    # workers.us.google.status.901 + workers.us.google        41.577825
    """

    if thresholds is None:
        thresholds = {
            'ex_workers.crawler_nodes.bing': 10,
            'papi.queue.web_insight': 110000.0,
            'proper.redis.requeue.standard.google': 2000.0,
            'redis.net.clients': 10500.0,
            'workers.us.google.status.901 + workers.us.google': 2.0
            }

    ans = pd.DataFrame(pd.np.zeros((len(df), len(thresholds) + 1)).astype(bool),
                       columns=[k + '__anomaly' for k in list(thresholds)] + ['any_anomaly'],
                       index=df.index)
    for dfk, ansk in zip(df.columns, ans.columns):
        ans[ansk] = df[dfk] > thresholds[dfk]
        ans['any_anomaly'] |= ans[ansk]
    return ans


def join_spans(spans):
    spans = list(spans)
    joined_spans = [list(spans[0])]
    for i, (start, stop) in enumerate(spans[1:]):
        if start > joined_spans[i][1]:
            joined_spans += [[start, stop]]
        else:
            joined_spans[i - 1][1] = stop
    return joined_spans


def plot_all(df=None, fillna_method='ffill', dropna=False, filename='time-series.html'):
    df = os.path.join(DATA_PATH, 'dd') if df is None else df
    df = clean_all(df, fillna_method=fillna_method, dropna=dropna).astype(float) if isinstance(df, str) else pd.DataFrame(df)

    anoms = is_anomalous(df)
    anom_spans = anoms['any_anomaly'].astype(int).diff().fillna(0)
    starts = list(df.index[anom_spans > 0])
    stops = list(df.index[anom_spans < 0])
    if len(stops) == len(starts) - 1:
        stops += [df.index.values[-1]]
    anom_spans = join_spans(join_spans(zip(starts, stops)))

    print(anom_spans)

    df['Num. Anomalous Monitors'] = (anoms[anoms.columns[:-1]].sum(axis=1) + .01)
    offline.plot(df.iplot(
        asFigure=True, xTitle='Date-Time', yTitle='Monitor Value', kind='scatter', logy=True,
        vspan=anom_spans),
        filename=filename,
        )
         
