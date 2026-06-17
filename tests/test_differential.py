import numpy as np
import pytest
from anndata import AnnData
from scipy.stats import ttest_ind, mannwhitneyu, kruskal, f_oneway, levene

from pychromvar.differential import (
    differential_deviations,
    differential_variability,
    _welch_anova,
)


def _make_dev(seed=0, n_cells=120, n_motif=5, n_groups=2):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, size=(n_cells, n_motif))
    raw = rng.normal(0, 1, size=(n_cells, n_motif))
    groups = np.array([f"g{i % n_groups}" for i in range(n_cells)])
    # inject a real group effect in motif 0 so tests are not all null
    g0 = groups == "g0"
    raw[g0, 0] += 1.5
    z[g0, 0] *= 2.0
    dev = AnnData(z.astype(np.float32))
    dev.layers["deviations"] = raw.astype(np.float32)
    dev.var_names = [f"motif_{i}" for i in range(n_motif)]
    dev.obs["cell_type"] = groups
    return dev, z, raw, groups


def _bh(p):
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    ranked = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, None, 1.0)
    return out


def test_welch_anova_matches_welch_ttest_for_two_groups():
    # For two groups Welch's ANOVA F == (Welch t)^2 and the p-values agree.
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 40)
    b = rng.normal(0.5, 2.0, 55)
    assert np.isclose(_welch_anova([a, b]),
                      ttest_ind(a, b, equal_var=False).pvalue, rtol=1e-9)


def test_differential_deviations_parametric_two_groups():
    dev, _, raw, groups = _make_dev(n_groups=2)
    res = differential_deviations(dev, "cell_type", parametric=True)

    g0 = groups == "g0"
    p_exp = np.array([
        ttest_ind(raw[g0, m], raw[~g0, m], equal_var=False).pvalue
        for m in range(raw.shape[1])
    ])
    assert np.allclose(res["p_value"].values, p_exp, rtol=1e-5)
    assert np.allclose(res["p_value_adjusted"].values, _bh(p_exp), rtol=1e-5)
    # the planted effect in motif_0 should be the most significant
    assert res["p_value"].values[0] == res["p_value"].values.min()


def test_differential_deviations_nonparametric_and_multigroup():
    dev, _, raw, groups = _make_dev(seed=1, n_groups=2)
    res = differential_deviations(dev, "cell_type", parametric=False)
    g0 = groups == "g0"
    p_exp = np.array([
        mannwhitneyu(raw[g0, m], raw[~g0, m], alternative="two-sided").pvalue
        for m in range(raw.shape[1])
    ])
    assert np.allclose(res["p_value"].values, p_exp, rtol=1e-5)

    dev3, _, raw3, groups3 = _make_dev(seed=2, n_groups=3)
    res3 = differential_deviations(dev3, "cell_type", parametric=False)
    codes = {g: i for i, g in enumerate(sorted(set(groups3)))}
    c = np.array([codes[g] for g in groups3])
    p_exp3 = np.array([
        kruskal(*[raw3[c == k, m] for k in range(3)]).pvalue
        for m in range(raw3.shape[1])
    ])
    assert np.allclose(res3["p_value"].values, p_exp3, rtol=1e-5)


def test_differential_variability_parametric_is_brown_forsythe():
    dev, z, _, groups = _make_dev(n_groups=2)
    res = differential_variability(dev, "cell_type", parametric=True)

    g0 = groups == "g0"
    # Brown-Forsythe == Levene with median centering == ANOVA on |x - median|
    p_exp = np.array([
        levene(z[g0, m], z[~g0, m], center="median").pvalue
        for m in range(z.shape[1])
    ])
    assert np.allclose(res["p_value"].values, p_exp, rtol=1e-5)
    # planted variance difference in motif_0 should be most significant
    assert res["p_value"].values[0] == res["p_value"].values.min()


def test_differential_variability_nonparametric():
    dev, _, _, groups = _make_dev(seed=4, n_groups=2)
    res = differential_variability(dev, "cell_type", parametric=False)

    # cross-check from the stored (float32) Z-scores the function actually reads
    z = np.asarray(dev.X, dtype=np.float64)
    g0 = groups == "g0"
    exp = []
    for m in range(z.shape[1]):
        md = np.empty(z.shape[0])
        md[g0] = np.abs(z[g0, m] - np.median(z[g0, m]))
        md[~g0] = np.abs(z[~g0, m] - np.median(z[~g0, m]))
        exp.append(kruskal(md[g0], md[~g0]).pvalue)
    assert np.allclose(res["p_value"].values, np.array(exp), rtol=1e-5)


def test_differential_accepts_vector_groups():
    dev, _, raw, groups = _make_dev(n_groups=2)
    by_name = differential_deviations(dev, "cell_type")
    by_vec = differential_deviations(dev, groups)
    assert np.allclose(by_name["p_value"].values, by_vec["p_value"].values,
                       equal_nan=True)
    with pytest.raises(ValueError):
        differential_deviations(dev, groups[:-1])  # wrong length
