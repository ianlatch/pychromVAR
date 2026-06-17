from typing import Union, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import chi2
from anndata import AnnData

from .compute_deviations import _resolve_n_jobs


def _row_sds(z: np.ndarray, na_rm: bool = True) -> np.ndarray:
    """Standard deviation (sample, ddof=1) of the deviation Z-scores across
    cells, per motif. ``z`` is cells x motifs. Mirrors ``src/utils.cpp::row_sds``:
    with ``na_rm`` a motif needs >= 2 finite values, else its SD is NaN.
    """
    if na_rm:
        n_finite = np.sum(~np.isnan(z), axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            sds = np.nanstd(z, axis=0, ddof=1)
        sds = np.where(n_finite >= 2, sds, np.nan)
        return sds
    return np.std(z, axis=0, ddof=1)


def _bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment, matching R ``p.adjust(method="BH")``
    (NaNs ignored, then restored)."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(~np.isnan(p))
    pv = p[idx]
    n = pv.size
    if n == 0:
        return out
    order = np.argsort(-pv)                       # descending
    ranks = np.arange(n, 0, -1)                   # n, n-1, ..., 1
    adj = np.minimum.accumulate(pv[order] * n / ranks)
    adj = np.clip(adj, None, 1.0)
    res = np.empty(n)
    res[order] = adj
    out[idx] = res
    return out


def compute_variability(deviations: Union[AnnData], bootstrap_error: bool = True,
                        bootstrap_samples: int = 1000,
                        bootstrap_quantiles: Tuple[float, float] = (0.025, 0.975),
                        na_rm: bool = True, seed: int = None,
                        n_jobs: int = -1) -> pd.DataFrame:
    """Compute the overall variability of each motif across cells.

    Faithful port of chromVAR's ``computeVariability``. Variability is the SD of
    a motif's deviation Z-scores across cells; the p-value tests whether that SD
    exceeds the null expectation of 1 via a chi-square statistic.

    Parameters
    ----------
    deviations : AnnData
        Output of :func:`compute_deviations`; ``.X`` holds the Z-scores.
    bootstrap_error : bool, optional
        Compute a bootstrap confidence interval for the variability, by default True.
    bootstrap_samples : int, optional
        Number of bootstrap resamples (over cells), by default 1000.
    bootstrap_quantiles : tuple, optional
        Lower/upper quantiles for the bootstrap interval, by default (0.025, 0.975).
    na_rm : bool, optional
        Ignore NaN Z-scores when computing SDs, by default True.
    seed : int, optional
        Seed for the bootstrap sampler, by default None.
    n_jobs : int, optional
        Worker threads for the bootstrap (-1 = all cores). Results are
        independent of this value, by default -1.

    Returns
    -------
    pandas.DataFrame
        Indexed by motif, with columns ``name``, ``variability``, ``p_value``,
        ``p_value_adj`` and (if ``bootstrap_error``) ``bootstrap_lower_bound``,
        ``bootstrap_upper_bound``.
    """
    z = np.asarray(deviations.X, dtype=np.float64)
    n_cells = z.shape[0]

    variability = _row_sds(z, na_rm)
    # chromVAR uses the total cell count for the chi-square df, not finite count.
    p_value = chi2.sf((n_cells - 1) * variability ** 2, df=n_cells - 1)
    p_value_adj = _bh_adjust(p_value)

    names = list(deviations.var_names)
    data = {
        "name": names,
        "variability": variability,
        "p_value": p_value,
        "p_value_adj": p_value_adj,
    }

    if bootstrap_error:
        if not (bootstrap_samples > 0):
            raise ValueError("bootstrap_samples must be positive")
        lo, hi = bootstrap_quantiles
        if not (0 < lo < hi < 1):
            raise ValueError("bootstrap_quantiles must satisfy 0 < lower < upper < 1")

        # Fast path: skip per-call NaN masking when the input is already finite.
        if na_rm and not np.isnan(z).any():
            def _sd(mat):
                return np.std(mat, axis=0, ddof=1)
        else:
            def _sd(mat):
                return _row_sds(mat, na_rm)

        # Independent RNG per bootstrap (spawned from one seed) keeps results
        # reproducible and identical regardless of the number of threads.
        child_seeds = np.random.SeedSequence(seed).spawn(bootstrap_samples)

        def _one(b):
            r = np.random.default_rng(child_seeds[b])
            idx = r.integers(0, n_cells, n_cells)     # resample cells w/ replacement
            return _sd(z[idx])

        boot = np.empty((bootstrap_samples, z.shape[1]))
        n_workers = _resolve_n_jobs(n_jobs)
        if n_workers == 1:
            for b in range(bootstrap_samples):
                boot[b] = _one(b)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                for b, res in enumerate(ex.map(_one, range(bootstrap_samples))):
                    boot[b] = res

        with np.errstate(invalid='ignore'):
            lower = np.nanquantile(boot, lo, axis=0)
            upper = np.nanquantile(boot, hi, axis=0)
        data["bootstrap_lower_bound"] = lower
        data["bootstrap_upper_bound"] = upper

    cols = ["name", "variability"]
    if bootstrap_error:
        cols += ["bootstrap_lower_bound", "bootstrap_upper_bound"]
    cols += ["p_value", "p_value_adj"]
    return pd.DataFrame(data, index=pd.Index(names, name=None))[cols]
