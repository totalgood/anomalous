from __future__ import print_function, unicode_literals, division, absolute_import
from future import standard_library
standard_library.install_aliases()  # noqa: Counter, OrderedDict, 
from builtins import *  # noqa:

import configparser
import logging
import logging.config
import os

from pugnlp.util import dict2obj


USER_HOME = os.path.expanduser("~")
PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
BIGDATA_PATH = os.path.join(os.path.dirname(__file__), 'bigdata')


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


# rename secrets.cfg.EXAMPLE_TEMPLATE -> secrets.cfg
# then edit secrets.cfg to include your actual credentials
secrets = configparser.RawConfigParser()
try:
    secrets.read(os.path.join(PROJECT_PATH, 'secrets.cfg'))
    secrets = secrets._sections
except IOError:
    logger.error('Unable to load/parse secrets.cfg file at "{}". Does it exist?'.format(os.path.join(PROJECT_PATH, 'secrets.cfg')))
    secrets = {}

secrets = dict2obj(secrets)
