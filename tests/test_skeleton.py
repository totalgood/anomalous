#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from python3.skeleton import fib

__author__ = "Hobson Lane"
__copyright__ = "AuthorityLabs"
__license__ = "none"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
