from typing import Union
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


def compute_deviations(data: Union[AnnData, MuData], threshold: float = 1.0,
                       chunk_size: int = 10000, n_jobs=-1) -> AnnData:
    """Compute bias-corrected deviations and deviation Z-scores.

    Faithful port of chromVAR's ``computeDeviations``. The returned object holds
    the deviation Z-scores in ``.X`` (chromVAR's ``deviationScores``) and the raw
    bias-corrected deviations in ``.layers['deviations']`` (chromVAR's
    ``deviations``). Per-motif QC matching chromVAR is stored in ``.var``:
    ``fractionMatches`` and ``fractionBackgroundOverlap``.

    Parameters
    ----------
    data : Union[AnnData, MuData]
        AnnData object with peak counts or MuData object with 'atac' modality.
    threshold : float, optional
        Minimum expected fragments; cell/motif entries whose expected count is
        below this are set to NaN (chromVAR's ``threshold``), by default 1.0.
    chunk_size : int, optional
        Number of cells processed per chunk, by default 10000.
    n_jobs : int, optional
        Accepted for backwards compatibility; currently unused, by default -1.

    Returns
    -------
    AnnData
        Deviations object: ``.X`` = Z-scores, ``.layers['deviations']`` = raw
        bias-corrected deviations.
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
    expectation_obs, expectation_var = compute_expectation(count=adata.X)

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

    for X, start, end in tqdm(adata.chunked_X(chunk_size), position=0,
                              leave=False, ncols=80, desc="cells"):
        eo = expectation_obs[start:end]
        obs_dev = _compute_deviations((motif_match, X, eo, expectation_var))

        # Online mean / sample-variance over background iterations (ddof=1, to
        # match chromVAR's `sd`), avoiding a full (n_bg, n_cells, n_motif) buffer.
        bg_sum = np.zeros((end - start, n_motifs), dtype=np.float64)
        bg_sumsq = np.zeros((end - start, n_motifs), dtype=np.float64)
        for i in range(n_bg_peaks):
            bg_idx = bg_peaks[:, i]
            # chromVAR sums counts AT each motif peak's background peak: reorder
            # the count (and expectation) columns, keep the motif annotation.
            bg_d = _compute_deviations(
                (motif_match, X[:, bg_idx], eo, expectation_var[:, bg_idx]))
            bg_sum += bg_d
            bg_sumsq += bg_d * bg_d

        mean_bg = bg_sum / n_bg_peaks
        var_bg = (bg_sumsq - n_bg_peaks * mean_bg ** 2) / (n_bg_peaks - 1)
        np.clip(var_bg, 0.0, None, out=var_bg)
        std_bg = np.sqrt(var_bg)

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


def compute_expectation(count: Union[np.array, sparse.csr_matrix]) -> np.array:
    """
    Compute expetation accessibility per peak and per cell by assuming
    identical read probability per peak for each cell with a sequencing
    depth matched to that cell observed sequencing depth

    Parameters
    ----------
    count : Union[np.array, sparse.csr_matrix]
        Count matrix containing raw accessibility data.

    Returns
    -------
    np.array, np.array
        Expectation matrix pair when multiplied gives
    """
    a = np.asarray(count.sum(0), dtype=np.float32).reshape((1, count.shape[1]))
    a /= a.sum()
    b = np.asarray(count.sum(1), dtype=np.float32).reshape((count.shape[0], 1))
    return b, a
