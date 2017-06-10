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

import time
import datetime
import json
import os
import sys
import re
import pickle
from collections import Mapping

import pandas as pd
import dateutil
import timestring
from sklearn.preprocessing import MinMaxScaler
from plotly import offline
import cufflinks  # noqa
import datadog as dd  # noqa
from dogapi.http import DogHttpApi

from pugnlp.futil import find_files, read_json
from pugnlp.util import dict2obj
from nlpia.data.loaders import read_csv

from .constants import logging, SECRETS
from .constants import DATA_PATH, DEFAULT_JSON_PATH, DEFAULT_DB_CSV_PATH, DEFAULT_META_PATH, DEFAULT_MODEL_PATH
from .constants import NAME_STRIP_CHRS, CFG

DATADOG_OPTIONS = SECRETS.datadog.__dict__

api = None
dd = dd


__author__ = "Hobson Lane"
__copyright__ = "AuthorityLabs"
__license__ = "none"

logger = logging.getLogger(__name__)


def dd_initialize(api_key=SECRETS.datadog.api_key, app_key=SECRETS.datadog.app_key):
    global api, dd
    if api is None:
        api = DogHttpApi(api_key=api_key, application_key=app_key, json_responses=False)
        dd.initialize(api_key=api_key, app_key=app_key)
    return dd, api


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


def parse_datetime_span(s, allow_none=True, default='past 24 hours'):
    """Parse a datetime string and extract any datetime or datetime ranges described within the text

    >>> parse_datetime_span('What time would your like to meet? 5:45 today in room B512?')
    (datetime.datetime(20... 5, 45), datetime.datetime(20...))
    >>> span = parse_datetime_span('24 hours ago')
    >>> span[1] - span[0]
    datetime.timedelta(1)
    >>> datetime.datetime.now() - span[1]
    datetime.timedelta(0, 0, ...)
    """
    s = s if allow_none else (default if s is None else s)
    if s is None:
        return s
    s = s.strip().strip("'")  # otherwise the quote "'24 hours ago'" will produce a time 1 hour ago.
    now = datetime.datetime.now()
    try:
        span = timestring.Range(s)
        if (span.end.date.toordinal() == datetime.date.today().toordinal() and
                span.start.date - span.end.date == datetime.timedelta(-1)):
            return now - datetime.timedelta(1), now
        elif (span.end.date - span.start.date).total_seconds() > 0:
            return span.start.date, span.end.date
        else:
            return span.end.date, span.start.date
    except timestring.TimestringInvalid:
        pass
    try:
        date = timestring.Date(s).date
        assert (now - date).total_seconds() > 0
        return date, now
    except timestring.TimestringInvalid:
        pass
    date = dateutil.parser.parse(s)
    assert (now - date).total_seconds() > 0
    return date, now


def argparse_datetime_span(parser, s, allow_none=True):
    """Parse a datetime string and extract any datetime or datetime ranges described within the text

    >>> argparse_datetime_span('What time would your like to meet? 5:45 today in room B512?')
    (<timestring.Date 2017-06-06 05:45:00 140134918475616>
    """
    try:
        return parse_datetime_span(s, allow_none=allow_none)
    except (timestring.TimestringInvalid, ValueError):
        parser.error("Unable to extract any datetimes from \"{}\"".format(s))
    except AssertionError:
        parser.error("Datetime(s) for the string \"{}\" are in the future.".format(s))
    return False


def read_series(file_or_path=None, i=0):
    """Convert a DataDog "series" json file into a Pandas Series"""
    file_or_path = file_or_path or DEFAULT_JSON_PATH
    js = read_json(file_or_path)
    ts = pd.Series(js['series'][i])
    return ts


def clean_dd_series(series):
    """Convert a DataDog "series" dictionary to a Pandas Series"""
    # t = pd.Series(pd.np.array(series['pointlist']).T[0],
    #               name='datetime')
    t, x = zip(*series['pointlist'])
    t = pd.np.array(t) / 1000.
    t = pd.np.array([datetime.datetime.fromtimestamp(ms) for ms in t])
    t = pd.to_datetime(t)
    metricname = series['display_name'].strip().strip(NAME_STRIP_CHRS).lower()
    hostname = series['scope'].strip().strip(NAME_STRIP_CHRS)
    hostname = hostname[4:] if hostname.startswith('host') else hostname
    hostname = hostname.strip().strip(NAME_STRIP_CHRS)

    name = ':'.join([s for s in [hostname, metricname] if len(s)])
    name = name.replace(' ', '')
    name = name[:48]

    ts = pd.Series(x, index=t, name=name)
    return ts


def clean_dd_df(file_or_path=None):
    """Load a DataDog json file and return a Pandas DataFrame of all the time series in the json list of dicts.

    Args:
      filepath (str): path to json file from DataDog dump

    Returns:
      pd.Series: Pandas Series with the timestamp as the index and a series name composed from the file path

    """
    file_or_path = file_or_path or os.path.join(DATA_PATH, 'dd', 'bing_nodes_online', 'day_1.json')
    df = pd.DataFrame()
    if isinstance(file_or_path, (list, dict)):
        js = file_or_path
    else:
        try:
            with (open(file_or_path, 'r') if isinstance(file_or_path, (str, bytes)) else file_or_path) as f:
                js = json.load(f)
        except IOError:
            js = json.loads(file_or_path)
    js = js.get('series', js) if hasattr(js, 'get') else js
    js = js if isinstance(js, list) else [js]
    for series in js:
        ts = clean_dd_series(series)
        df[ts.name] = ts
    return df


def load_all(dirpath=os.path.join(DATA_PATH, 'dd')):
    files = find_files(dirpath, ext='json')
    df = pd.DataFrame()
    for f in files:
        df = df.append(clean_dd_df(f['path']))
    return df


def clean_dd_all(df=None, fillna_method='ffill', dropna=True, consolidate_index=False):
    """Undersample, interpolate, or impute values to fill in NaNs"""
    df = os.path.join(DATA_PATH, 'dd') if df is None else df
    df = load_all(df).astype(float) if isinstance(df, str) else pd.DataFrame(df)

    df.index = pd.to_datetime(df.index.values)
    df.sort_index(inplace=True)
    df = df.reindex()

    if consolidate_index:
        df = df[~df.index.duplicated(keep='last')]
        df = df.reindex()

    df.fillna(method=fillna_method, inplace=True, axis=0)
    if dropna is True:
        df.dropna(inplace=True)
    else:
        df.fillna(value=0, inplace=True)
    return df


def scale_all(df=None, fillna_method='ffill', dropna=False):
    df = os.path.join(DATA_PATH, 'dd') if df is None else df
    df = clean_dd_all(df, fillna_method=fillna_method, dropna=dropna).astype(float) if isinstance(df, str) else pd.DataFrame(df)

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
            'proper.redis.seattle.1.ala.bs:redis.net.clients': 10500.0,
            'workers.us.google.status.901+workers.us.google.s': 2.0
            }

    ans = pd.DataFrame(pd.np.zeros((len(df), len(thresholds) + 1)).astype(bool),
                       columns=[k + '__anomaly' for k in list(thresholds)] + ['any_anomaly'],
                       index=df.index)
    for dfk, ansk in zip(df.columns, ans.columns):
        if dfk in thresholds:
            ans[ansk] = df[dfk] > thresholds[dfk]
            ans['any_anomaly'] |= ans[ansk]
        else:
            logger.warn('No threshold defined for {}'.format(dfk))
            print(dfk)
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
    df = clean_dd_all(df, fillna_method=fillna_method, dropna=dropna).astype(float) if isinstance(df, str) else pd.DataFrame(df)

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
    return df


def get_dd_hosts(pattern='', regex=''):
    """Get a list of host names filtered by a DataDog query pattern (default='') and regex (default='')

    >>> get_dd_hosts()
    ['balancer.1.api.authoritylabs.com',
    ...
    'worker.us.js.test.05.api.superioritylabs.com']
    >>> get_dd_hosts(regex='[.]test[.]')
    ['worker.us.js.test.01.api.superioritylabs.com',
     'worker.us.js.test.02.api.superioritylabs.com',
     'worker.us.js.test.03.api.superioritylabs.com',
     'worker.us.js.test.04.api.superioritylabs.com',
     'worker.us.js.test.05.api.superioritylabs.com']
    >>> get_dd_hosts(regex='^[.]test[.]$')
    []
    """
    global dd, api
    dd, api = dd_initialize()
    hosts = api.search('hosts:{}'.format(pattern))['hosts']
    if regex:
        regex = re.compile(regex)
        return [h for h in hosts if regex.search(h)]
    return hosts


def toordinal(obj, allow_none=True):
    """Convert a str, timestring.Date, datetime.date, or datetime.datetime to an ordinal (integer days since 1 AD)

    >>> toordinal('Christmas in 1984')
    724435
    >>> toordinal(datetime.datetime(2017, 6, 1, 2, 34, 56))
    736481
    """
    if allow_none and obj is None:
        return None
    if isinstance(obj, int):
        return obj
    if hasattr(obj, 'toordinal'):
        return obj.toordinal()
    if hasattr(obj, 'to_pydatetime'):
        return toordinal(obj.to_pydatetime())()
    if hasattr(obj, 'to_datetime'):
        return toordinal(obj.to_datetime())()
    if hasattr(obj, 'date') and not callable(obj.date):
        return toordinal(obj.date)
    if isinstance(obj, str):
        return toordinal(timestring.Date(obj))
    raise ValueError('Unable to coerce {} into an int ordinal (datetime.toordinal() days since 01 AD'.format(obj))


def timestamp(obj, allow_none=True):
    """Convert a str, timestring.Date, datetime.date, or datetime.datetime to an ordinal (float seconds since 1 AD)

    >>> timestamp('Christmas in 1984')
    timestamp('Christmas in 1984')
    >>> 1970 + timestamp(datetime.datetime(2017, 6, 1, 2, 34, 56)) / 365.25 / 24. / 3600.
    2017.41519304383...
    """
    if allow_none and obj is None:
        return None
    if isinstance(obj, float):
        return obj
    if hasattr(obj, 'timestamp'):
        if callable(obj.timestamp):
            return obj.timestamp()
        return timestamp(obj.timestamp)
    if hasattr(obj, 'to_pydatetime'):
        return timestamp(obj.to_pydatetime())()
    if hasattr(obj, 'to_datetime'):
        return timestamp(obj.to_datetime())()
    if hasattr(obj, 'date') and not callable(obj.date):
        return timestamp(obj.date)
    if isinstance(obj, str):
        return timestamp(timestring.Date(obj))
    raise ValueError('Unable to coerce {} into a timestamp (datetime.timestamp() float seconds since 1970-01-01 00:00:00'.format(obj))


def get_dd_metric(metric_name=None, servers=None, start=None, end=None):
    global dd, api
    dd, api = dd_initialize()  # noqa
    end = int(timestamp(end or time.time()))
    start = int(timestamp(start or end - 3600. * 24.))
    query = metric_name + r'{*}by{host}'  # 'system.cpu.idle{*}by{host}'
    # series = dd.api.Metric.query(start=now - 3600 * 24, end=now, query=query)
    logger.debug('start={}, end={}, query={}'.format(start, end, query))
    series = dd.api.Metric.query(start=start, end=end, query=query)
    return clean_dd_df(series)


def update_config_dict(config, args, skip_nones=True):
    """Update configuration (from configparser) dict with args dict (from argparser) ignoring None in args

    `args` values override `config` values unless the args value is None and `skip_nones` is True

    >>> update_config_dict({'x': 1, 'y': None, 'zz': 'zzz...'}, {'x': None, 'zz': 'She is woke!'})
    {'x': 1, 'y': None, 'zz': 'She is woke!'}
    """
    args = dict([(k, v) for (k, v) in args.items() if (not skip_nones or v is not None or k not in config)])
    config.update(args)
    return config


def update_config(cfg, args, skip_nones=True):
    """Update configuration (from configparser) object with args object (from argparser)

    SEE: argparse_config package
    `args` values override `config` values unless the args value is None and `skip_nones` is True

    >>> update_config(dict2obj({'x': 1, 'y': None, 'zz': 'zzz...'}),
    ...               dict2obj({'x': None, 'zz': 'She is woke!'})
    ...              ).__dict__
    {'x': 1, 'y': None, 'zz': 'She is woke!'}
    """
    cfg = cfg if isinstance(cfg, Mapping) else cfg.__dict__
    args = args if isinstance(args, Mapping) else args.__dict__
    return dict2obj(update_config_dict(cfg, args, skip_nones=skip_nones))


def update_db(metric_names=CFG.metrics, start=None, end=None, db=None, drop=False):
    """Query DataDog to retrieve all the metrics for the time span indicated and save them to db.csv.gz"""
    end = pd.to_datetime(datetime.datetime.now() if end is None else end)
    dbpath = db
    if not isinstance(dbpath, str):
        dbpath = DEFAULT_DB_CSV_PATH
        if drop:
            db = pd.DataFrame()
    try:
        db = read_csv(dbpath)
    except IOError:
        db = pd.DataFrame()

    for metric_name in metric_names:
        if start is None:
            if len(db) and not db.index.max().isnull():
                db_end = pd.to_datetime(db.index.max())
            else:
                day_ago = pd.to_datetime(datetime.datetime.now() - datetime.timedelta(1))
            start = pd.Series([db_end, day_ago]).min()
        oneday = start + datetime.timedelta(1)
        while oneday - end < datetime.timedelta(1):
            logger.info("Querying Datadog for {} to {}".format(start, oneday))
            df = get_dd_metric(metric_name=metric_name, start=start, end=oneday)
            logger.info("Retrieved {} metrics".format(df.shape))
            db = db.append(df)
            start = oneday
            oneday = start + datetime.timedelta(1)
    db.sort_index(inplace=True)
    db.reindex()
    if isinstance(dbpath, str):
        db.to_csv(dbpath, compression='gzip')
        logger.info("Saved db.shape={} to {}".format(db.shape, dbpath))
        logger.debug(db.describe())
    return db


def get_meta():
    global dd, api
    dd, api = dd_initialize()  # noqa
    return api.search('')


def update_meta(meta=None, drop=False):
    """Query DataDog to retrieve all the host and metric names and save them to meta.json"""
    empty_meta = {'hosts': [], 'metrics': []}
    metapath = meta
    if not isinstance(metapath, str):
        metapath = DEFAULT_META_PATH
        if drop:
            meta = empty_meta
    try:
        with open(metapath, 'rt') as f:
            meta = json.load(f)
    except IOError:
        meta = empty_meta
    new_meta = get_meta()
    for k in ['hosts', 'metrics']:
        meta[k] = sorted(set(meta[k]).union(set(new_meta[k])))
    with open(metapath, 'wt') as f:
        json.dump(meta, f, indent=4)
    return meta


def retrain_model(path=DEFAULT_MODEL_PATH):
    with open(path, 'rt') as f:
        model = pickle.load(f)
    with open(path, 'rt') as f:
        pickle.dump(model, f)
