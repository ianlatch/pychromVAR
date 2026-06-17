import numpy as np
from anndata import AnnData

from pychromvar.compute_deviations import (
    compute_deviations,
    compute_expectation,
    _compute_deviations,
)


def _chromvar_oracle(counts, motif_match, bg_peaks, threshold=1.0):
    """Independent re-implementation of chromVAR's ``compute_deviations_single``
    (compute_deviations.R:451-537), written in cell x peak orientation, used as
    the cross-implementation oracle.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n_cells, n_peaks = counts.shape
    n_motif = motif_match.shape[1]
    n_bg = bg_peaks.shape[1]

    frag_per_cell = counts.sum(axis=1)                 # fragments per sample
    expectation = counts.sum(axis=0) / counts.sum()    # per-peak read fraction

    z = np.full((n_cells, n_motif), np.nan)
    dev = np.full((n_cells, n_motif), np.nan)
    frac_matches = np.zeros(n_motif)
    frac_overlap = np.full(n_motif, np.nan)

    for m in range(n_motif):
        S = np.flatnonzero(motif_match[:, m])
        frac_matches[m] = S.size / n_peaks
        if S.size == 0:
            continue

        observed = counts[:, S].sum(axis=1)
        expected = frag_per_cell * expectation[S].sum()
        obs_dev = (observed - expected) / expected

        sampled_dev = np.zeros((n_bg, n_cells))
        overlaps = np.zeros(n_bg)
        for r in range(n_bg):
            bgset = bg_peaks[S, r]
            sampled = counts[:, bgset].sum(axis=1)
            sampled_exp = frag_per_cell * expectation[bgset].sum()
            sampled_dev[r] = (sampled - sampled_exp) / sampled_exp
            overlaps[r] = np.isin(bgset, S).sum()
        frac_overlap[m] = overlaps.mean() / S.size

        mean_bg = sampled_dev.mean(axis=0)
        sd_bg = sampled_dev.std(axis=0, ddof=1)        # chromVAR uses sd()
        normdev = obs_dev - mean_bg
        zscore = normdev / sd_bg

        fail = expected < threshold
        normdev[fail] = np.nan
        zscore[fail] = np.nan
        dev[:, m] = normdev
        z[:, m] = zscore

    return z, dev, frac_matches, frac_overlap


def _build_adata(seed=0, n_cells=8, n_peaks=10, n_motif=3, n_bg=12):
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 6, size=(n_cells, n_peaks)).astype(np.float32)
    # guarantee every peak has at least one fragment
    counts[0, :] += 1
    motif_match = rng.integers(0, 2, size=(n_peaks, n_motif)).astype(np.uint8)
    # guarantee every motif annotates at least one peak
    for m in range(n_motif):
        if motif_match[:, m].sum() == 0:
            motif_match[0, m] = 1
    bg_peaks = rng.integers(0, n_peaks, size=(n_peaks, n_bg))

    adata = AnnData(counts)
    adata.varm['motif_match'] = motif_match
    adata.varm['bg_peaks'] = bg_peaks
    adata.uns['motif_name'] = [f"motif_{i}" for i in range(n_motif)]
    return adata, counts, motif_match, bg_peaks


def test_compute_expectation():
    count = np.array([[1, 0, 1], [0, 1, 1]])
    b, a = compute_expectation(count)

    # a: per-peak fraction of reads (sums to 1); b: reads per cell
    assert np.allclose(a, np.array([[0.25, 0.25, 0.5]]))
    assert np.allclose(b, np.array([[2.0], [2.0]]))
    # outer-style product reproduces the expected count matrix
    expected = b.dot(a)
    assert np.allclose(expected, np.array([[0.5, 0.5, 1.0], [0.5, 0.5, 1.0]]))


def test_compute_deviations_matches_chromvar_oracle():
    adata, counts, motif_match, bg_peaks = _build_adata()
    out = compute_deviations(adata)

    z_exp, dev_exp, fm_exp, fo_exp = _chromvar_oracle(counts, motif_match, bg_peaks)

    assert np.allclose(out.X, z_exp, rtol=1e-4, atol=1e-4, equal_nan=True)
    assert np.allclose(out.layers['deviations'], dev_exp,
                       rtol=1e-4, atol=1e-4, equal_nan=True)
    assert np.allclose(out.var['fractionMatches'].values, fm_exp)
    assert np.allclose(out.var['fractionBackgroundOverlap'].values, fo_exp,
                       equal_nan=True)


def test_compute_deviations_output_structure():
    adata, _, _, _ = _build_adata()
    out = compute_deviations(adata)

    assert out.shape == (adata.n_obs, adata.varm['motif_match'].shape[1])
    assert 'deviations' in out.layers
    assert list(out.var_names) == list(adata.uns['motif_name'])
    assert list(out.obs_names) == list(adata.obs_names)


def test_compute_deviations_threshold_masks_low_expected():
    adata, _, _, _ = _build_adata()
    # An absurdly high threshold should mask every entry to NaN.
    out = compute_deviations(adata, threshold=1e9)
    assert np.isnan(out.X).all()
    assert np.isnan(out.layers['deviations']).all()


def test_compute_expectation_norm_and_group():
    import scipy.sparse as sp
    rng = np.random.default_rng(5)
    counts = rng.integers(1, 7, size=(12, 6)).astype(np.float64)
    depth = counts.sum(1)

    # norm=True: a[peak] = sum_cell count[cell,peak] / depth[cell]
    b, a = compute_expectation(counts, norm=True)
    a_exp = (counts / depth.reshape(-1, 1)).sum(0)
    assert np.allclose(a.ravel(), a_exp)
    assert np.allclose(b.ravel(), depth)

    # sparse matches dense for the norm path
    b_s, a_s = compute_expectation(sp.csr_matrix(counts), norm=True)
    assert np.allclose(a_s.ravel(), a_exp)

    # group, no norm: mean over groups of within-group peak fraction
    group = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    _, a_g = compute_expectation(counts, group=group)
    g0 = counts[:6].sum(0) / counts[:6].sum()
    g1 = counts[6:].sum(0) / counts[6:].sum()
    assert np.allclose(a_g.ravel(), (g0 + g1) / 2)


def test_compute_deviations_accepts_precomputed_expectation():
    adata, counts, motif_match, bg_peaks = _build_adata()
    exp = compute_expectation(adata.X)
    out_default = compute_deviations(adata)
    out_explicit = compute_deviations(adata, expectation=exp)
    assert np.allclose(out_default.X, out_explicit.X, equal_nan=True)


def test_compute_deviations_threaded_matches_serial():
    # larger bg count + multiple chunks so the threaded path is exercised
    adata, _, _, _ = _build_adata(seed=3, n_cells=50, n_motif=4, n_bg=40)
    serial = compute_deviations(adata, n_jobs=1, chunk_size=16)
    threaded = compute_deviations(adata, n_jobs=4, chunk_size=16)
    # in-order accumulation keeps the threaded result bit-identical
    assert np.allclose(serial.X, threaded.X, rtol=0, atol=0, equal_nan=True)
    assert np.allclose(serial.layers['deviations'], threaded.layers['deviations'],
                       rtol=0, atol=0, equal_nan=True)


def test_internal_deviation_helper():
    count = np.array([[1, 0, 1], [0, 1, 1]])
    b, a = compute_expectation(count)
    motif_match = np.array([[1, 1], [0, 1], [1, 0]], dtype=np.uint8)

    dev = _compute_deviations((motif_match, count, b, a))
    # observed - expected, divided by expected, per (cell, motif)
    observed = count.dot(motif_match)
    expected = b.dot(a.dot(motif_match))
    ref = (observed - expected) / expected
    assert np.allclose(dev, ref)
