#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pytest
from anomalous.chasedown import clean_df
from anomalous.constants import DEFAULT_JSON_PATH

__author__ = "Hobson Lane"
__copyright__ = "AuthorityLabs"
__license__ = "none"


def test_clean_df():
    df = clean_df(DEFAULT_JSON_PATH)
    assert df.shape == (288, 1)
    with pytest.raises(IOError):
        clean_df('nonexistent path')
