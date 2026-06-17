import numpy as np
import pytest
from anndata import AnnData

from pychromvar.filtering import (
    get_fragments_per_peak,
    get_fragments_per_sample,
    get_total_fragments,
    filter_peaks,
    filter_samples,
)


def _adata(counts, peak_names=None):
    a = AnnData(np.asarray(counts, dtype=np.float32))
    if peak_names is not None:
        a.var_names = peak_names
    return a


def test_fragment_getters():
    counts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # cells x peaks
    a = _adata(counts)
    assert np.allclose(get_fragments_per_peak(a), [5, 7, 9])      # sum over cells
    assert np.allclose(get_fragments_per_sample(a), [6, 15])      # sum over peaks
    assert get_total_fragments(a) == 21


def test_filter_peaks_min_fragments():
    counts = np.array([[0, 2, 0, 5],
                       [0, 1, 0, 4]], dtype=np.float32)
    # peaks chosen far apart so non-overlapping does nothing
    names = ["chr1-100-200", "chr1-300-400", "chr1-500-600", "chr1-700-800"]
    a = _adata(counts, names)
    keep = filter_peaks(a, min_fragments_per_peak=1, non_overlapping=True,
                        ix_return=True)
    assert np.array_equal(keep, [1, 3])  # peaks 0 and 2 are empty


def test_filter_peaks_removes_overlaps_keeping_larger():
    # peaks 0 and 1 overlap on chr1; peak 1 has more fragments -> keep peak 1.
    counts = np.array([[1, 9, 3],
                       [1, 9, 2]], dtype=np.float32)
    names = ["chr1-100-250", "chr1-200-350", "chr2-100-200"]
    a = _adata(counts, names)
    keep = filter_peaks(a, non_overlapping=True, ix_return=True)
    assert np.array_equal(keep, [1, 2])

    # returned object form
    filtered = filter_peaks(a, non_overlapping=True)
    assert list(filtered.var_names) == ["chr1-200-350", "chr2-100-200"]


def test_filter_peaks_disjoint_returns_all():
    counts = np.ones((2, 3), dtype=np.float32)
    names = ["chr1-100-200", "chr1-300-400", "chr2-100-200"]
    a = _adata(counts, names)
    keep = filter_peaks(a, non_overlapping=True, ix_return=True)
    assert np.array_equal(keep, [0, 1, 2])


def test_filter_samples_depth_and_in_peaks():
    # 4 cells x 2 peaks; explicit depth column
    counts = np.array([[10, 10],   # 20 in peaks
                       [1, 1],      # 2 in peaks
                       [400, 400],  # 800 in peaks
                       [5, 5]], dtype=np.float32)
    a = _adata(counts)
    a.obs["depth"] = [40, 1000, 1000, 10]   # total library size per cell

    # in-peaks fraction = [0.5, 0.002, 0.8, 1.0]
    keep = filter_samples(a, min_depth=30, min_in_peaks=0.4, ix_return=True)
    # cell0: depth40>=30 & 0.5>=0.4 -> keep; cell1: frac too low; cell2: keep;
    # cell3: depth10<30 -> drop
    assert np.array_equal(keep, [0, 2])


def test_filter_samples_defaults_run():
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 50, size=(20, 5)).astype(np.float32)
    a = _adata(counts)
    a.obs["depth"] = counts.sum(1) * rng.uniform(1.0, 3.0, size=20)
    out = filter_samples(a)  # defaults estimated from data, should not error
    assert out.n_obs <= a.n_obs
