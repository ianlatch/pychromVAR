import numpy as np
from anndata import AnnData
from scipy.stats import chi2

from pychromvar.compute_variability import compute_variability, _bh_adjust


def _bh_oracle(p):
    """Plain Benjamini-Hochberg, used to cross-check _bh_adjust."""
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, None, 1.0)
    return out


def _make_dev(seed=0, n_cells=200, n_motif=6):
    rng = np.random.default_rng(seed)
    # give motifs differing variability so p-values span a range
    scales = np.linspace(0.5, 3.0, n_motif)
    z = rng.normal(0.0, 1.0, size=(n_cells, n_motif)) * scales
    dev = AnnData(z.astype(np.float32))
    dev.var_names = [f"motif_{i}" for i in range(n_motif)]
    return dev, z


def test_variability_matches_chromvar_formula():
    dev, z = _make_dev()
    res = compute_variability(dev, bootstrap_error=False)

    n = z.shape[0]
    var_exp = np.std(z, axis=0, ddof=1)
    p_exp = chi2.sf((n - 1) * var_exp ** 2, df=n - 1)
    padj_exp = _bh_oracle(p_exp)

    assert np.allclose(res["variability"].values, var_exp, rtol=1e-4)
    assert np.allclose(res["p_value"].values, p_exp, rtol=1e-4)
    assert np.allclose(res["p_value_adj"].values, padj_exp, rtol=1e-4)
    assert list(res["name"]) == list(dev.var_names)
    # no bootstrap columns when disabled
    assert "bootstrap_lower_bound" not in res.columns


def test_bh_adjust_matches_oracle():
    p = np.array([0.001, 0.04, 0.03, 0.5, 0.2, 0.9])
    assert np.allclose(_bh_adjust(p), _bh_oracle(p))
    # adjusted p-values are >= raw and bounded by 1
    assert np.all(_bh_adjust(p) >= p - 1e-12)
    assert np.all(_bh_adjust(p) <= 1.0)


def test_bootstrap_ci_structure_and_reproducible():
    dev, _ = _make_dev()
    r1 = compute_variability(dev, bootstrap_samples=200, seed=7)
    r2 = compute_variability(dev, bootstrap_samples=200, seed=7)

    for col in ["bootstrap_lower_bound", "bootstrap_upper_bound"]:
        assert col in r1.columns
    # lower <= upper, and the point estimate sits inside a wide-ish interval
    assert np.all(r1["bootstrap_lower_bound"].values <= r1["bootstrap_upper_bound"].values)
    # same seed -> identical results
    assert np.allclose(r1["bootstrap_lower_bound"].values, r2["bootstrap_lower_bound"].values)
    assert np.allclose(r1["bootstrap_upper_bound"].values, r2["bootstrap_upper_bound"].values)


def test_bootstrap_independent_of_n_jobs():
    dev, _ = _make_dev()
    serial = compute_variability(dev, bootstrap_samples=200, seed=11, n_jobs=1)
    threaded = compute_variability(dev, bootstrap_samples=200, seed=11, n_jobs=4)
    # per-bootstrap RNGs make the result identical regardless of thread count
    assert np.allclose(serial["bootstrap_lower_bound"].values,
                       threaded["bootstrap_lower_bound"].values, equal_nan=True)
    assert np.allclose(serial["bootstrap_upper_bound"].values,
                       threaded["bootstrap_upper_bound"].values, equal_nan=True)


def test_variability_handles_nan_zscores():
    dev, z = _make_dev()
    # mask some entries to NaN (as the deviation threshold filter would)
    z = z.copy()
    z[0:5, 0] = np.nan
    dev2 = AnnData(z.astype(np.float32))
    dev2.var_names = list(dev.var_names)

    res = compute_variability(dev2, bootstrap_error=False, na_rm=True)
    expected0 = np.nanstd(z[:, 0], ddof=1)
    assert np.isclose(res["variability"].values[0], expected0, rtol=1e-4)
    assert not np.isnan(res["variability"].values).any()
