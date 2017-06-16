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
from pandas import np
import dateutil
import timestring
from sklearn.preprocessing import MinMaxScaler
from plotly import offline
import cufflinks  # noqa
import datadog as dd  # noqa
from dogapi.http import DogHttpApi
from tqdm import tqdm

from pugnlp.futil import find_files, read_json
from pugnlp.util import dict2obj
from nlpia.data.loaders import read_csv

from .constants import logging, SECRETS
from .constants import DEFAULT_JSON_DIR, DEFAULT_JSON_PATH, DEFAULT_DB_CSV_PATH, DEFAULT_META_PATH, DEFAULT_MODEL_PATH, DEFAULT_HUMAN_PATH
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


def normalize_metric_name(display_name, scope='*', max_len=80):
    """Combine host FQDN and metric display_name to create unique "key" for each metric"""
    if '{' in display_name or ':' in display_name:
        name = display_name.strip().lower()
    else:
        metricname = display_name.strip().strip(NAME_STRIP_CHRS).lower()

        hostname = scope.strip().strip(NAME_STRIP_CHRS)
        hostname = hostname[4:] if hostname.startswith('host') else hostname
        hostname = hostname.strip().strip(NAME_STRIP_CHRS)

        name = ':'.join([s for s in [hostname, metricname] if len(s)])

    name = re.sub(r'\s', '', name)
    name = name[:max_len]
    return name


def dd_time_to_datetime(t):
    t = pd.np.array(t) / 1000.
    t = pd.np.array([datetime.datetime.fromtimestamp(ms) for ms in t])
    t = pd.to_datetime(t)
    return t


def clean_dd_series(series, name=None):
    """Convert a DataDog "series" dictionary to a Pandas Series"""
    # t = pd.Series(pd.np.array(series['pointlist']).T[0],
    #               name='datetime')
    t, x = zip(*series['pointlist'])

    t = dd_time_to_datetime(t)
    if name is None:
        name = normalize_metric_name(series['display_name'], scope=series['scope'])

    ts = pd.Series(x, index=t, name=name)
    return ts


def clean_dd_df(file_or_path=None, name=None):
    """Load a DataDog json file and return a Pandas DataFrame of all the time series in the json list of dicts.

    Args:
      filepath (str): path to json file from DataDog dump

    Returns:
      pd.Series: Pandas Series with the timestamp as the index and a series name composed from the file path

    """
    file_or_path = file_or_path or DEFAULT_JSON_PATH
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
        ts = clean_dd_series(series, name=name)
        df[ts.name] = ts
    return df


def load_all(dirpath=DEFAULT_JSON_DIR):
    files = find_files(dirpath, ext='json')
    df = pd.DataFrame()
    for f in files:
        df = df.append(clean_dd_df(f['path']))
    return df


def clean_dd_all(df=None, fillna_method='ffill', dropna=False, consolidate_index=False):
    """Undersample, interpolate, or impute values to fill in NaNs"""
    df = DEFAULT_JSON_DIR if df is None else df
    df = load_all(df).astype(float) if isinstance(df, str) else pd.DataFrame(df)

    df.index = pd.to_datetime(df.index.values)
    df.sort_index(inplace=True)
    df.groupby(df.index).mean()
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
    df = DEFAULT_JSON_DIR if df is None else df
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
        thresholds = [(q['query'], q['threshold']) for q in CFG.queries if q['query'] in df.columns]
    queries, values = zip(*thresholds)
    lows, highs = zip(*values)

    ans = pd.DataFrame(pd.np.zeros((len(df), len(queries) + 1)).astype(bool),
                       columns=['anomaly__<{}&>{}__{}'.format(low, high, query) for query, low, high in zip(queries, lows, highs)] + ['anomaly__any'],
                       index=df.index)
    for dfk, ansk, (low, high) in zip(queries, ans.columns, values):
        high = float(high) if isinstance(high, str) else high
        low = float(low) if isinstance(low, str) else low
        high = np.nan if pd.isnull(high) else high
        low = np.nan if pd.isnull(low) else low
        if dfk in queries:
            if not pd.isnull(high):
                ans[ansk] |= df[dfk] > high
            if not pd.isnull(low):
                ans[ansk] |= df[dfk] < low
            ans['anomaly__any'] |= ans[ansk]
        else:
            logger.error('No threshold defined for {}'.format(dfk))
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


def ask_if_anomalous(new_spans, human_labels_path=DEFAULT_HUMAN_PATH):
    """Console input by user to confirm or deny anomalous timespans"""
    try:
        df = read_csv(human_labels_path)
    except IOError:
        df = pd.DataFrame(columns='start end human_label'.split())

    columns = df.columns
    print('Anomalous Time Spans Detected in Past 24 hours:')
    print(pd.DataFrame(new_spans, columns=columns[:2]))
    print('Refer to the time-series.html plot for this time period to determine whether these time spans were truly anomalous.')
    human_labels = pd.np.zeros(len(new_spans), dtype=int)
    for i, (start, end) in enumerate(new_spans):
        print("{}: {} to {}".format(i, start, end))
        ans = input("Is the time span above anomalous [Y]/N/Yall/Nall? ")
        if re.match(r'y|Y|Yes|YES|yes|yep|yup', ans):
            human_labels[i] = 1
        elif ans.lower().strip().endswith('all'):
            if ans.lower().strip() == 'yall':
                human_labels[i:] = np.array([1] * len(human_labels[i:]))
                break
            elif ans.lower().strip() == 'nall':
                break

    dfnew = pd.DataFrame(new_spans, columns=df.columns[:2])
    dfnew[df.columns[-1]] = human_labels
    df = df.append(dfnew, ignore_index=True)
    df.to_csv(human_labels_path)
    return df


def plot_all(df=None, fillna_method='ffill', dropna=False, filename='time-series.html'):
    df = DEFAULT_JSON_DIR if df is None else df
    df = clean_dd_all(df, fillna_method=fillna_method, dropna=dropna).astype(float) if isinstance(df, str) else pd.DataFrame(df)

    anoms = is_anomalous(df)
    anom_spans = anoms['anomaly__any'].astype(int).diff().fillna(0)
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


def plot_predictions(df=None, fillna_method='ffill', dropna=False, filename='time-series.html'):
    """Plot predicted anomalies and queries that likely triggered them in an HTML file and print the anomalous time spans

    >>> plot_predictions(df=db[datetime.datetime.now() - datetime.timedelta(1):])
    """
    df = DEFAULT_JSON_DIR if df is None else df
    df = clean_dd_all(df, fillna_method=fillna_method, dropna=dropna).astype(float) if isinstance(df, str) else pd.DataFrame(df)

    rf = pickle.load(open(DEFAULT_MODEL_PATH, 'rb'))
    # df = clean_dd_all(df)
    # thresholds = [(q['query'], q['threshold']) for q in CFG.queries if q['query'] in df.columns]
    predictions = rf.predict(df.values)
    anoms = pd.DataFrame(predictions,
                         columns=[q['query'] for q in CFG.queries if q['query'] in df.columns] + ['anomaly__any'],
                         index=df.index.values)
    # anoms = is_anomalous(df)  # manually-determined thresholds on queries from Chase
    anoms['anomaly__any'].iloc[0] = 0
    anoms['anomaly__any'].iloc[-1] = 0

    anom_spans = anoms['anomaly__any'].astype(int).diff().fillna(0)
    starts = list(df.index[anom_spans > 0])
    stops = list(df.index[anom_spans < 0])
    if len(stops) == len(starts) - 1:
        stops += [df.index.values[-1]]
    if len(starts) > 0 and len(stops) > 0:
        anom_spans = join_spans(zip(starts, stops))
    else:
        anom_spans = []

    df = df[[c for c in anoms.columns if c in df.columns]]
    df['Num. Anomalous Monitors'] = (anoms[anoms.columns[:-1]].sum(axis=1) + .01)
    df.columns = [normalize_metric_name(c, max_len=64) for c in df.columns]
    offline.plot(df.iplot(
        asFigure=True, xTitle='Date-Time', yTitle='Monitor Value', kind='scatter', logy=True,
        vspan=anom_spans),
        filename=filename,
        )

    ask_if_anomalous(anom_spans)

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
    if isinstance(obj, (int, float)):
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


def get_dd_metric(name=None, servers=None, start=None, end=None):
    global dd, api
    dd, api = dd_initialize()  # noqa
    end = int(timestamp(end or time.time()))
    start = int(timestamp(start or end - 3600 * 24))
    query = name + r'{*}by{host}'  # 'system.cpu.idle{*}by{host}'
    # series = dd.api.Metric.query(start=now - 3600 * 24, end=now, query=query)
    logger.debug('start={}, end={}, query={}'.format(start, end, query))
    series = dd.api.Metric.query(start=start, end=end, query=query)
    return clean_dd_df(series)


def get_dd_metrics(names=None, servers=None, start=None, end=None):
    metric_names = CFG.metrics if names is None else sorted(names)
    df_metrics = pd.DataFrame()
    start0 = start
    for i, metric_name in enumerate(tqdm(metric_names)):
        logger.info('Requesting metric {}/{} {} for {} to {}'.format(
            i, len(metric_names), metric_name, start0, end))
        start = start0
        oneday = start + datetime.timedelta(1)
        while oneday - end < datetime.timedelta(1):
            logger.debug("Querying Datadog for {} to {}".format(start, oneday))
            df = get_dd_metric(metric_name, start=start, end=oneday)
            logger.debug("Retrieved {} metrics".format(df.shape))
            df_metrics = df_metrics.append(df)
            start = oneday
            oneday = start + datetime.timedelta(1)
    return df_metrics


def get_dd_query(query, start=None, end=None):
    global dd, api
    dd, api = dd_initialize()  # noqa
    end = int(timestamp(end or time.time()))
    start = int(timestamp(start or (end - 3600 * 24)))
    # series = dd.api.Metric.query(start=now - 3600 * 24, end=now, query=query)
    logger.debug('start={}, end={}, query={}'.format(start, end, query))
    series = dd.api.Metric.query(start=start, end=end, query=query)
    return clean_dd_df(series, name=query)


def get_dd_queries(queries=None, start=None, end=None):
    queries = CFG.queries if queries is None else list(queries)
    end = int(timestamp(end or time.time()))
    start = int(timestamp(start or end - 3600 * 24))
    dfall = pd.DataFrame()
    start0 = start
    for i, q in enumerate(tqdm(queries)):
        query = q['query']
        logger.info('Requesting query {}/{} {} for {} to {}'.format(
            i + 1, len(queries), query, start, end))
        start = start0
        oneday = start + 3600 * 24
        while oneday - end < 3600 * 24:
            logging.debug("Querying Datadog for {} to {}".format(start, oneday))
            df = get_dd_query(query=query, start=start, end=oneday)
            logging.debug("Retrieved {} monitor query values".format(df.shape))
            dfall = dfall.append(df)
            start = oneday
            oneday = start + 3600 * 24
    return dfall


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


def update_db(db=None, metric_names=CFG.metrics, start=None, end=None, drop=False, save=True):
    """Query DataDog to retrieve all the metrics for the time span indicated and save them to db.csv.gz"""
    dbpath = db
    if not isinstance(dbpath, str):
        dbpath = DEFAULT_DB_CSV_PATH
    if drop:
        db = pd.DataFrame()
    else:
        try:
            db = read_csv(dbpath)
        except IOError:
            db = pd.DataFrame()

    end = pd.to_datetime(datetime.datetime.now() if end is None else end)
    if start is None:
        if len(db) and not db.index.max().isnull():
            db_end = pd.to_datetime(db.index.max())
        else:
            day_ago = pd.to_datetime(datetime.datetime.now() - datetime.timedelta(1))
        start = pd.Series([db_end, day_ago]).min()
    start = pd.to_datetime(start)

    df = get_dd_metrics(CFG.metrics, start=start, end=end)
    if drop:
        db = df
    else:
        db = db.append(df)
    db.sort_index(inplace=True)
    db = db.reindex()
    db = db.groupby([db.index]).mean()
    db.sort_index(inplace=True)
    db = db.reindex()
    if isinstance(dbpath, str) and save:
        db.to_csv(dbpath, compression='gzip')
        print(db.columns)
        logger.info("Saved db.shape={} to {}".format(db.shape, dbpath))
        logger.debug(db.describe())

    df = get_dd_queries(CFG.queries, start=start, end=end)
    db = db.append(df)
    db.sort_index(inplace=True)
    db.reindex()
    if isinstance(dbpath, str):
        logger.info('Saving appended database of historical metrics (shaped {}) to a single CSV file, so this may take a while (minutes)...'.format(
            df.shape))
        db.to_csv(dbpath, compression='gzip')
        logger.info(db.columns)
        logger.info("Saved db.shape={} to {}".format(db.shape, dbpath))
        logger.debug(db.describe())
    return db


def get_meta():
    global dd, api
    dd, api = dd_initialize()  # noqa
    return api.search('')


def update_meta(meta=None, drop=False):
    """Query DataDog to retrieve all the host and metric names and save them to meta.json"""
    empty_meta = {'hosts': [], 'metrics': [], 'monitors': {}}
    metapath = meta
    meta = empty_meta
    if not isinstance(metapath, str):
        metapath = DEFAULT_META_PATH
    if not drop:
        try:
            with open(metapath, 'rt') as f:
                meta.update(json.load(f))
        except IOError:
            pass
    new_meta = get_meta()
    monitors = dd.api.monitors.Monitor.get_all(group_states=['all'])
    monitor_names = [monitor_dict['name'] for monitor_dict in monitors]
    monitor_queries = [monitor_dict['query'] for monitor_dict in monitors]
    new_meta['monitors'] = dict(zip(monitor_names, monitor_queries))
    for k in ['hosts', 'metrics']:
        meta[k] = sorted(set(meta[k]).union(set(new_meta[k])))
    for k in ['monitors']:
        meta[k].update(new_meta[k])
    with open(metapath, 'wt') as f:
        json.dump(meta, f, indent=4)
    return meta


def retrain_model(path=DEFAULT_MODEL_PATH):
    with open(path, 'rt') as f:
        model = pickle.load(f)
    with open(path, 'rt') as f:
        pickle.dump(model, f)
