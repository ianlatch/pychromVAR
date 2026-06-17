from typing import Union, Sequence
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, kruskal, f_oneway, f as _f_dist
from anndata import AnnData

from .compute_variability import _bh_adjust


def _resolve_groups(deviations: AnnData, groups: Union[str, Sequence]) -> np.ndarray:
    """Return a per-cell group label array from a column name or a vector."""
    if isinstance(groups, str):
        if groups in deviations.obs.columns:
            return np.asarray(deviations.obs[groups].values)
        raise ValueError(f"'{groups}' is not a column in .obs")
    g = np.asarray(groups)
    if g.shape[0] != deviations.n_obs:
        raise ValueError(
            "groups must be a column name in .obs or a vector of length n_obs")
    return g


def _welch_anova(samples: Sequence[np.ndarray]) -> float:
    """Welch's heteroscedastic one-way ANOVA p-value, matching R's
    ``oneway.test(var.equal = FALSE)``. ``samples`` is a list of 1-D arrays."""
    k = len(samples)
    n = np.array([s.size for s in samples], dtype=np.float64)
    if np.any(n < 2):
        return np.nan
    m = np.array([s.mean() for s in samples])
    v = np.array([s.var(ddof=1) for s in samples])
    if np.any(v <= 0):
        return np.nan
    w = n / v
    sw = w.sum()
    xbar = (w * m).sum() / sw
    num = (w * (m - xbar) ** 2).sum() / (k - 1)
    tmp = (((1 - w / sw) ** 2) / (n - 1)).sum()
    denom = 1 + (2 * (k - 2) / (k ** 2 - 1)) * tmp
    fstat = num / denom
    df1 = k - 1
    df2 = (k ** 2 - 1) / (3 * tmp)
    return float(_f_dist.sf(fstat, df1, df2))


def _split_dropna(x: np.ndarray, codes: np.ndarray, n_levels: int):
    """Split x by group code, dropping NaN values within each group."""
    out = []
    for lvl in range(n_levels):
        vals = x[codes == lvl]
        vals = vals[~np.isnan(vals)]
        out.append(vals)
    return out


def differential_deviations(deviations: AnnData, groups: Union[str, Sequence],
                            alternative: str = "two-sided",
                            parametric: bool = True) -> pd.DataFrame:
    """Test whether bias-corrected deviations differ between groups of cells.

    Port of chromVAR's ``differentialDeviations`` (uses the ``deviations``
    assay). Two groups: Welch t-test or Mann-Whitney; more: Welch ANOVA or
    Kruskal-Wallis. Returns a DataFrame indexed by motif with ``p_value`` and
    ``p_value_adjusted``.

    Parameters
    ----------
    deviations : AnnData
        Output of :func:`compute_deviations`.
    groups : str or sequence
        ``.obs`` column name or a per-cell vector of group labels.
    alternative : str, optional
        Two-group only: 'two-sided', 'less', or 'greater'.
    parametric : bool, optional
        Use parametric tests. Default True.
    """
    g = pd.Categorical(_resolve_groups(deviations, groups))
    codes = g.codes
    n_levels = len(g.categories)
    if n_levels < 2:
        raise ValueError("groups must contain at least two levels")

    inputs = np.asarray(deviations.layers['deviations'], dtype=np.float64)
    n_motifs = inputs.shape[1]
    p_val = np.full(n_motifs, np.nan)

    for m in range(n_motifs):
        split = _split_dropna(inputs[:, m], codes, n_levels)
        if any(s.size < 1 for s in split):
            continue
        try:
            if parametric and n_levels == 2:
                p_val[m] = ttest_ind(split[0], split[1], equal_var=False,
                                     alternative=alternative).pvalue
            elif parametric:
                p_val[m] = _welch_anova(split)
            elif n_levels == 2:
                p_val[m] = mannwhitneyu(split[0], split[1],
                                        alternative=alternative).pvalue
            else:
                p_val[m] = kruskal(*split).pvalue
        except ValueError:
            p_val[m] = np.nan

    return pd.DataFrame(
        {"p_value": p_val, "p_value_adjusted": _bh_adjust(p_val)},
        index=pd.Index(list(deviations.var_names)))


def differential_variability(deviations: AnnData, groups: Union[str, Sequence],
                             parametric: bool = True) -> pd.DataFrame:
    """Test whether variability differs between groups of cells.

    Port of chromVAR's ``differentialVariability``: a Brown-Forsythe test on the
    Z-scores (ANOVA, or Kruskal-Wallis if non-parametric, on the absolute
    deviations from each group's median). Returns a DataFrame indexed by motif
    with ``p_value`` and ``p_value_adjusted``.

    Parameters
    ----------
    deviations : AnnData
        Output of :func:`compute_deviations`; uses the Z-scores in ``.X``.
    groups : str or sequence
        ``.obs`` column name or a per-cell vector of group labels.
    parametric : bool, optional
        Use the parametric Brown-Forsythe test. Default True.
    """
    g = pd.Categorical(_resolve_groups(deviations, groups))
    codes = g.codes
    n_levels = len(g.categories)
    if n_levels < 2:
        raise ValueError("groups must contain at least two levels")

    inputs = np.asarray(deviations.X, dtype=np.float64)
    n_motifs = inputs.shape[1]
    p_val = np.full(n_motifs, np.nan)

    for m in range(n_motifs):
        x = inputs[:, m]
        finite = ~np.isnan(x)
        # absolute deviation from the per-group median (Brown-Forsythe)
        median_diff = np.empty_like(x)
        ok = True
        for lvl in range(n_levels):
            sel = finite & (codes == lvl)
            if sel.sum() < 1:
                ok = False
                break
            median_diff[sel] = np.abs(x[sel] - np.median(x[sel]))
        if not ok:
            continue
        split = [median_diff[finite & (codes == lvl)] for lvl in range(n_levels)]
        try:
            if parametric:
                p_val[m] = f_oneway(*split).pvalue
            else:
                p_val[m] = kruskal(*split).pvalue
        except ValueError:
            p_val[m] = np.nan

    return pd.DataFrame(
        {"p_value": p_val, "p_value_adjusted": _bh_adjust(p_val)},
        index=pd.Index(list(deviations.var_names)))
