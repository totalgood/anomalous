from __future__ import print_function, unicode_literals, division, absolute_import
from future import standard_library
standard_library.install_aliases()  # noqa: Counter, OrderedDict, 
from builtins import *  # noqa:

import configparser
import logging
import logging.config
import os
import json

from pugnlp.util import dict2obj


USER_HOME = os.path.expanduser("~")
PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
BIGDATA_PATH = os.path.join(os.path.dirname(__file__), 'bigdata')

DEFAULT_SECRETS_PATH = os.path.join(PROJECT_PATH, 'secrets.cfg')
DEFAULT_JSON_DIR = os.path.join(DATA_PATH, 'dd')
DEFAULT_JSON_PATH = os.path.join(DEFAULT_JSON_DIR, 'bing_nodes_online', 'day_1.json')

DEFAULT_DB_DIR = os.path.join(DATA_PATH, 'db')
DEFAULT_DB_CSV_FILENAME = 'db.csv.gz'
DEFAULT_DB_CSV_PATH = os.path.join(DEFAULT_DB_DIR, DEFAULT_DB_CSV_FILENAME)
DEFAULT_ANOMALIES_CSV_FILENAME = 'anomalies.csv.gz'
DEFAULT_ANOMALIES_CSV_PATH = os.path.join(DEFAULT_DB_DIR, DEFAULT_DB_CSV_FILENAME)
DEFAULT_META_FILENAME = 'meta.json'
DEFAULT_META_PATH = os.path.join(DEFAULT_DB_DIR, DEFAULT_META_FILENAME)
DEFAULT_MODEL_FILENAME = 'model.pkl'
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_DB_DIR, DEFAULT_MODEL_FILENAME)
DEFAULT_HUMAN_FILENAME = 'human-labeled-time-spans.csv'
DEFAULT_HUMAN_PATH = os.path.join(DEFAULT_DB_DIR, DEFAULT_HUMAN_FILENAME)  # start, end, 0/1 (1=anomalous)
DEFAULT_CONFIG_FILENAME = 'config.cfg'
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_DB_DIR, DEFAULT_CONFIG_FILENAME)

NAME_STRIP_CHRS = '\t\n\r :-=+!._$%#@[]'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'django': {
            'format': 'django: %(message)s',
        },
        u'basic': {
            u'format': u'%(asctime)s | %(name)15s:%(lineno)3s:%(funcName)15s | %(levelname)7s | %(message)s',
        }
    },

    'handlers': {
        'logging.handlers.SysLogHandler': {
            'level': 'DEBUG',
            'class': 'logging.handlers.SysLogHandler',
            'facility': 'local7',
            'formatter': 'django',
            'address': '/dev/log',
        },
        u'console': {
            u'class': u'logging.StreamHandler',
            u'level': u'DEBUG',
            u'formatter': u'basic',
            u'stream': u'ext://sys.stdout',
        },
    },

    'loggers': {
        'loggly': {
            'handlers': [u'console', 'logging.handlers.SysLogHandler'],
            'propagate': True,
            'format': 'django: %(message)s',
            'level': 'DEBUG',
        },
    },
}

logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)


def parse_config(path=DEFAULT_CONFIG_PATH, section=None, eval_keys=['metrics', 'queries']):
    """ Pares a cfg file, retrieving the requested section as a dictionary

    Args:
        section (str): name of the section you'd like to extract
        eval_keys (list of str): Parser will try to evaluate strings in the config variables for the indicated eval_keys
    """
    configreader = configparser.RawConfigParser()
    try:
        configreader.read(path)
        configdict = configreader._sections
        configdict = configdict[section] if section else configdict
    except IOError:
        logger.error('Unable to load/parse .cfg file at "{}". Does it exist?'.format(
            path))
        configdict = {}
    for k in eval_keys:
        if k in configdict:
            try:
                configdict[k] = eval(configdict[k], {'__builtins__': None}, {})
            except:
                pass
    return dict2obj(configdict)


# these should be overridden by command line args or non-default config file paths
CFG = parse_config(DEFAULT_CONFIG_PATH, section='chasedown')
SECRETS = parse_config(DEFAULT_SECRETS_PATH, section=None)
if os.path.isfile(DEFAULT_META_PATH):
    with open(DEFAULT_META_PATH) as f:
        META = json.load(f)
else:
    META = {'hosts': [], 'metrics': [], 'monitors': {}}

# str treated as *str* globstar filter
if isinstance(CFG.metrics, str):
    CFG.metrics = [s for s in META['metrics'] if CFG.metrics in s]
elif hasattr(CFG.metrics, 'match') and hasattr(CFG.metrics, 'pattern'):
    CFG.metrics = [s for s in META['metrics'] if CFG.metrics.match(s)]
