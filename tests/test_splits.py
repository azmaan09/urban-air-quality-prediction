"""Tests for chronological splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.splits import create_rolling_origin_splits, create_temporal_split


def test_temporal_split_order_and_sizes():
    ts = pd.Series(pd.date_range("2015-01-01", periods=1000, freq="h", tz="UTC"))
    split = create_temporal_split(ts, 0.7, 0.15, 0.15)
    assert len(split.train_idx) + len(split.valid_idx) + len(split.test_idx) == 1000
    assert ts.iloc[split.train_idx].max() <= ts.iloc[split.valid_idx].min()
    assert ts.iloc[split.valid_idx].max() <= ts.iloc[split.test_idx].min()


def test_no_index_overlap():
    ts = pd.Series(pd.date_range("2015-01-01", periods=500, freq="h", tz="UTC"))
    split = create_temporal_split(ts, 0.7, 0.15, 0.15)
    s_train, s_valid, s_test = set(split.train_idx), set(split.valid_idx), set(split.test_idx)
    assert s_train.isdisjoint(s_valid)
    assert s_train.isdisjoint(s_test)
    assert s_valid.isdisjoint(s_test)


def test_rolling_origin_yields_expanding_train():
    ts = pd.Series(pd.date_range("2015-01-01", periods=20000, freq="h", tz="UTC"))
    sizes = []
    for tr, va in create_rolling_origin_splits(ts, n_splits=3, min_train_hours=5000, valid_hours=1000):
        sizes.append(len(tr))
        assert ts.iloc[tr].max() <= ts.iloc[va].min()
    assert sizes == sorted(sizes)
    assert len(sizes) == 3


def test_ratio_sum_validation():
    ts = pd.Series(pd.date_range("2015-01-01", periods=200, freq="h", tz="UTC"))
    with pytest.raises(ValueError):
        create_temporal_split(ts, 0.5, 0.5, 0.5)
