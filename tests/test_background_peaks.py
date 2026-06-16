import numpy as np
import pytest
from anndata import AnnData
from scipy.stats import norm as _norm_dist

from pychromvar.preprocessing import (
    get_bg_peaks,
    _bg_bin_structure,
    _sample_bg_peaks,
)


def test_bin_structure_invariants():
    rng = np.random.default_rng(0)
    n = 200
    intensity = rng.normal(2.0, 0.5, size=n)
    bias = rng.uniform(0.3, 0.7, size=n)
    bs, w = 8, 0.1

    bin_membership, bin_density, bin_p = _bg_bin_structure(intensity, bias, w, bs)

    # every peak assigned to a valid bin; densities account for all peaks
    assert bin_membership.shape == (n,)
    assert bin_membership.min() >= 0 and bin_membership.max() < bs * bs
    assert bin_density.shape == (bs * bs,)
    assert bin_density.sum() == n
    assert np.array_equal(bin_density, np.bincount(bin_membership, minlength=bs * bs))

    # bin_p is the Gaussian distance kernel: symmetric, diagonal = dnorm(0,0,w)
    assert bin_p.shape == (bs * bs, bs * bs)
    assert np.allclose(bin_p, bin_p.T)
    assert np.allclose(np.diag(bin_p), _norm_dist.pdf(0.0, 0.0, w))
    # off-diagonal kernel values never exceed the peak at distance 0
    assert bin_p.max() <= _norm_dist.pdf(0.0, 0.0, w) + 1e-12


def test_sampler_matches_theoretical_weights():
    # Hand-built bin layout (3 bins): peaks 0,1 in bin 0; peak 2 in bin 1;
    # peak 3 in bin 2. Verify draw frequencies match the chromVAR weighting
    # p(candidate k) proportional to bin_p[bin(k), target] / density(bin(k)).
    bin_membership = np.array([0, 0, 1, 2])
    bin_density = np.array([2, 1, 1])
    dist = np.array([[0.0, 1.0, 2.0],
                     [1.0, 0.0, 1.5],
                     [2.0, 1.5, 0.0]])
    bin_p = _norm_dist.pdf(dist, 0.0, 0.6)

    niterations = 40000
    rng = np.random.default_rng(123)
    out = _sample_bg_peaks(bin_membership, bin_density, bin_p, niterations, rng)

    assert out.shape == (4, niterations)

    # peaks 0 and 1 share bin 0 -> identical candidate distribution
    p = bin_p[bin_membership, 0] / bin_density[bin_membership]
    p /= p.sum()
    pooled = out[[0, 1], :].ravel()
    freq = np.bincount(pooled, minlength=4) / pooled.size
    assert np.allclose(freq, p, atol=0.01)


def test_get_bg_peaks_reproducible_and_valid():
    rng = np.random.default_rng(1)
    n_cells, n_peaks = 30, 120
    counts = rng.integers(1, 8, size=(n_cells, n_peaks)).astype(np.float32)
    adata = AnnData(counts)
    adata.var['gc_bias'] = rng.uniform(0.3, 0.7, size=n_peaks)

    get_bg_peaks(adata, niterations=25, seed=42)
    first = adata.varm['bg_peaks'].copy()

    assert first.shape == (n_peaks, 25)
    assert first.min() >= 0 and first.max() < n_peaks

    # same seed -> identical sampling
    get_bg_peaks(adata, niterations=25, seed=42)
    assert np.array_equal(first, adata.varm['bg_peaks'])


def test_get_bg_peaks_rejects_empty_peak():
    counts = np.ones((5, 4))
    counts[:, 2] = 0  # one peak with no fragments anywhere
    adata = AnnData(counts)
    adata.var['gc_bias'] = np.array([0.4, 0.5, 0.6, 0.5])

    with pytest.raises(ValueError):
        get_bg_peaks(adata, niterations=5, seed=0)
