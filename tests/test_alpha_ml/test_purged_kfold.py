"""Tests for PurgedKFold cross-validator."""

import numpy as np
import pandas as pd
import pytest

from alpha_ml.validation.purged_kfold import PurgedKFold


@pytest.fixture
def sample_index() -> pd.DatetimeIndex:
    """250-day date index — enough for 5 folds of 50 bars each."""
    return pd.date_range("2020-01-01", periods=250, freq="B")


class TestPurgedKFold:
    def test_n_splits_attribute(self) -> None:
        pkf = PurgedKFold(n_splits=5)
        assert pkf.n_splits == 5

    def test_produces_n_folds(self, sample_index: pd.DatetimeIndex) -> None:
        pkf = PurgedKFold(n_splits=5)
        folds = list(pkf.split(sample_index))
        assert len(folds) == 5

    def test_test_indices_cover_all_samples(self, sample_index: pd.DatetimeIndex) -> None:
        """All indices must appear in exactly one test fold."""
        pkf = PurgedKFold(n_splits=5)
        all_test: list[int] = []
        for _, test_idx in pkf.split(sample_index):
            all_test.extend(list(test_idx))
        # All indices covered
        assert len(all_test) == len(sample_index)
        # No duplicates
        assert len(all_test) == len(set(all_test))

    def test_test_indices_non_overlapping(self, sample_index: pd.DatetimeIndex) -> None:
        pkf = PurgedKFold(n_splits=5)
        test_sets: list[set[int]] = []
        for _, test_idx in pkf.split(sample_index):
            test_sets.append(set(test_idx))
        for i in range(len(test_sets)):
            for j in range(i + 1, len(test_sets)):
                assert test_sets[i].isdisjoint(test_sets[j])

    def test_train_and_test_disjoint(self, sample_index: pd.DatetimeIndex) -> None:
        pkf = PurgedKFold(n_splits=5)
        for train_idx, test_idx in pkf.split(sample_index):
            assert set(train_idx).isdisjoint(set(test_idx))

    def test_purging_removes_adjacent_train_samples(self, sample_index: pd.DatetimeIndex) -> None:
        """With purge_window=10, no train index should be within 10 of test fold start."""
        purge = 10
        pkf = PurgedKFold(n_splits=5, purge_window=purge)
        for train_idx, test_idx in pkf.split(sample_index):
            test_start = int(min(test_idx))
            # All train indices must be at least purge_window away from test_start
            close = [i for i in train_idx if 0 <= test_start - i < purge]
            assert len(close) == 0, f"Purging failed: {close} within {purge} of test_start={test_start}"

    def test_embargo_removes_post_test_samples(self, sample_index: pd.DatetimeIndex) -> None:
        """With embargo=5, no train index should be within 5 bars after test fold end."""
        embargo = 5
        pkf = PurgedKFold(n_splits=5, embargo=embargo)
        for train_idx, test_idx in pkf.split(sample_index):
            test_end = int(max(test_idx))
            # Train indices just after test_end (within embargo) must be absent
            close_after = [i for i in train_idx if 0 < i - test_end <= embargo]
            assert len(close_after) == 0, f"Embargo failed: {close_after} within {embargo} of test_end={test_end}"

    def test_returns_numpy_arrays(self, sample_index: pd.DatetimeIndex) -> None:
        pkf = PurgedKFold(n_splits=5)
        for train_idx, test_idx in pkf.split(sample_index):
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
