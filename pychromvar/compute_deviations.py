from typing import Union
import os
from concurrent.futures import ThreadPoolExecutor
from anndata import AnnData
from mudata import MuData
import numpy as np
from scipy import sparse
import logging
from tqdm.auto import tqdm

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def _resolve_n_jobs(n_jobs) -> int:
    """Translate the n_jobs convention (-1 / None -> all cores) to a worker count."""
    if n_jobs is None or n_jobs < 0:
        return os.cpu_count() or 1
    return max(1, int(n_jobs))


def _imap_bounded(executor, fn, n, batch):
    """Map ``fn`` over ``range(n)`` yielding results in submission order, with at
    most ``batch`` tasks in flight so only ``batch`` results are held at once."""
    for s in range(0, n, batch):
        yield from executor.map(fn, range(s, min(s + batch, n)))




def compute_deviations(data: Union[AnnData, MuData], threshold: float = 1.0,
                       expectation=None, chunk_size: int = 10000,
                       n_jobs=-1) -> AnnData:
    """Compute bias-corrected deviations and deviation Z-scores.

    Port of chromVAR's ``computeDeviations``. Returns an AnnData with Z-scores in
    ``.X``, raw bias-corrected deviations in ``.layers['deviations']``, and
    per-motif QC (``fractionMatches``, ``fractionBackgroundOverlap``) in ``.var``.

    Parameters
    ----------
    data : AnnData or MuData
        Peak counts (MuData uses the 'atac' modality).
    threshold : float, optional
        Entries with expected fragments below this are set to NaN. Default 1.0.
    expectation : tuple, optional
        Precomputed ``(b, a)`` from :func:`compute_expectation`; computed if None.
    chunk_size : int, optional
        Cells per chunk. Default 10000.
    n_jobs : int, optional
        Worker threads for the background loop (-1 = all cores). Default -1.
    """
    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError(
            "Expected AnnData or MuData object with 'atac' modality")
    # check if the object contains bias in Anndata.varm
    assert "bg_peaks" in adata.varm, \
        "Cannot find background peaks in the input object, please first run get_bg_peaks!"

    motif_match = adata.varm['motif_match']
    bg_peaks = adata.varm['bg_peaks']
    n_bg_peaks = bg_peaks.shape[1]
    n_motifs = motif_match.shape[1]

    logging.info('computing expectation reads per cell and peak...')
    if expectation is None:
        expectation_obs, expectation_var = compute_expectation(count=adata.X)
    else:
        expectation_obs, expectation_var = expectation

    # Per-motif QC (cell-independent) ----------------------------------------
    # fractionMatches: fraction of peaks annotated to each motif.
    tf_count = np.asarray(motif_match.sum(axis=0)).reshape(-1).astype(np.float64)
    fraction_matches = tf_count / adata.n_vars

    # fractionBackgroundOverlap: how often a motif peak's background peak is also
    # a motif peak (chromVAR compute_deviations_single, the `bg_overlap` term).
    overlap = np.zeros(n_motifs, dtype=np.float64)
    for i in range(n_bg_peaks):
        bg_motif_match = motif_match[bg_peaks[:, i], :]
        overlap += np.asarray((motif_match * bg_motif_match).sum(axis=0)).reshape(-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        fraction_bg_overlap = overlap / (n_bg_peaks * tf_count)

    # Expected fraction of reads per motif, for the threshold filter.
    peakfrac_motif = np.asarray(expectation_var @ motif_match).reshape(-1)

    logging.info('computing observed + bg motif deviations...')
    dev = np.zeros((adata.n_obs, n_motifs), dtype=np.float32)
    z = np.zeros((adata.n_obs, n_motifs), dtype=np.float32)

    n_workers = _resolve_n_jobs(n_jobs)
    executor = ThreadPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
    try:
        for X, start, end in tqdm(adata.chunked_X(chunk_size), position=0,
                                  leave=False, ncols=80, desc="cells"):
            eo = expectation_obs[start:end]
            obs_dev = _compute_deviations((motif_match, X, eo, expectation_var))

            def _bg_dev(i):
                bg_idx = bg_peaks[:, i]
                # chromVAR sums counts AT each motif peak's background peak:
                # reorder the count (and expectation) columns, keep the motif
                # annotation.
                return _compute_deviations(
                    (motif_match, X[:, bg_idx], eo, expectation_var[:, bg_idx]))

            # The sparse matmul releases the GIL, so threads parallelise it;
            # results are consumed in submission order so the output stays
            # bit-identical to the serial path.
            if executor is None:
                bg_iter = (_bg_dev(i) for i in range(n_bg_peaks))
            else:
                bg_iter = _imap_bounded(executor, _bg_dev, n_bg_peaks, n_workers)

            # Welford's online algorithm for the per-(cell, motif) mean and
            # sample variance (ddof=1, to match chromVAR's `sd`) over background
            # iterations. Single-pass (no (n_bg, n_cells, n_motif) buffer) and
            # numerically stable: each M2 increment equals delta^2 (count-1)/count
            # >= 0, so M2 never goes negative and no clipping is needed.
            mean_bg = np.zeros((end - start, n_motifs), dtype=np.float64)
            m2_bg = np.zeros((end - start, n_motifs), dtype=np.float64)
            count = 0
            for bg_d in bg_iter:
                count += 1
                delta = bg_d - mean_bg
                mean_bg += delta / count
                m2_bg += delta * (bg_d - mean_bg)

            std_bg = np.sqrt(m2_bg / (count - 1))

            normdev = obs_dev - mean_bg
            with np.errstate(invalid='ignore', divide='ignore'):
                zscore = normdev / std_bg

            # threshold filter: expected < threshold -> NaN
            expected = np.asarray(eo).reshape(-1, 1) * peakfrac_motif.reshape(1, -1)
            fail = expected < threshold
            normdev = normdev.astype(np.float32)
            zscore = zscore.astype(np.float32)
            normdev[fail] = np.nan
            zscore[fail] = np.nan

            dev[start:end, :] = normdev
            z[start:end, :] = zscore
    finally:
        if executor is not None:
            executor.shutdown()

    out = AnnData(z)
    out.layers['deviations'] = dev
    out.obs_names = adata.obs_names
    out.var_names = adata.uns['motif_name']
    out.var['fractionMatches'] = fraction_matches
    out.var['fractionBackgroundOverlap'] = fraction_bg_overlap
    return out


def _compute_deviations(arguments):
    motif_match, count, expectation_obs, expectation_var = arguments
    ### motif_match: n_var x n_motif
    ### count, exp: n_obs x n_var
    observed = count.dot(motif_match)
    expected = expectation_obs.dot(expectation_var.dot(motif_match))
    if sparse.issparse(observed):
        observed = observed.todense()
    if sparse.issparse(expected):
        expected = expected.todense()
    observed = np.asarray(observed, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    out = np.zeros(expected.shape, dtype=np.float64)
    np.divide(observed - expected, expected, out=out, where=expected != 0)
    return out


def compute_expectation(count: Union[np.array, sparse.csr_matrix],
                        norm: bool = False, group=None) -> np.array:
    """Expected accessibility per cell and peak (chromVAR's ``computeExpectations``).

    Returns a ``(b, a)`` pair: ``b`` = fragments per cell, ``a`` = per-peak
    expected read fraction; ``b @ a`` reconstructs the expected count matrix.

    Parameters
    ----------
    count : ndarray or csr_matrix
        Count matrix (cells x peaks).
    norm : bool, optional
        Weight all cells equally rather than by depth. Not recommended for
        single-cell data. Default False.
    group : optional
        Per-cell group labels; expectation becomes the mean across groups of the
        within-group per-peak fraction.
    """
    n_cells, n_peaks = count.shape
    b = np.asarray(count.sum(1), dtype=np.float32).reshape((n_cells, 1))

    def _norm_fraction(sub, depth):
        # sum over cells of count[cell, peak] / depth[cell]
        if sparse.issparse(sub):
            scaled = sparse.diags(1.0 / depth) @ sub
            return np.asarray(scaled.sum(0), dtype=np.float64).reshape(-1)
        return (np.asarray(sub, dtype=np.float64) /
                depth.reshape(-1, 1)).sum(0)

    if group is None:
        if norm:
            depth = np.asarray(b, dtype=np.float64).reshape(-1)
            a = _norm_fraction(count, depth).reshape((1, n_peaks))
        else:
            a = np.asarray(count.sum(0), dtype=np.float32).reshape((1, n_peaks))
            a /= a.sum()
    else:
        group = np.asarray(group)
        if group.shape[0] != n_cells:
            raise ValueError("group must be a vector of length n_cells")
        levels = np.unique(group)
        depth = np.asarray(b, dtype=np.float64).reshape(-1)
        mat = np.zeros((n_peaks, levels.size), dtype=np.float64)
        for i, lvl in enumerate(levels):
            ix = np.flatnonzero(group == lvl)
            sub = count[ix]
            if norm:
                mat[:, i] = _norm_fraction(sub, depth[ix])
            else:
                mat[:, i] = (np.asarray(sub.sum(0), dtype=np.float64).reshape(-1)
                             / sub.sum())
        a = mat.mean(axis=1).reshape((1, n_peaks))

    return b, a
