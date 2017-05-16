#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pytest
from anomalous.chasedown import clean_df
from anomalous.constants import DATA_PATH

__author__ = "Hobson Lane"
__copyright__ = "AuthorityLabs"
__license__ = "none"


def test_clean_df():
    df = clean_df(os.path.join(DATA_PATH, 'dd', 'bing_nodes_online', 'day_1.json'))
    assert df.shape == (288, 1)
    with pytest.raises(IOError):
        clean_df('nonexistent path')
