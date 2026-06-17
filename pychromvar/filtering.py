from typing import Union
import re
import numpy as np
from anndata import AnnData
from mudata import MuData


def _get_adata(data: Union[AnnData, MuData]) -> AnnData:
    if isinstance(data, AnnData):
        return data
    if isinstance(data, MuData) and "atac" in data.mod:
        return data.mod["atac"]
    raise TypeError("Expected AnnData or MuData object with 'atac' modality")


def get_fragments_per_peak(data: Union[AnnData, MuData]) -> np.ndarray:
    """Total fragments in each peak (summed across cells)."""
    adata = _get_adata(data)
    return np.asarray(adata.X.sum(axis=0)).reshape(-1)


def get_fragments_per_sample(data: Union[AnnData, MuData]) -> np.ndarray:
    """Total fragments in each cell/sample (summed across peaks)."""
    adata = _get_adata(data)
    return np.asarray(adata.X.sum(axis=1)).reshape(-1)


def get_total_fragments(data: Union[AnnData, MuData]) -> float:
    """Total fragments in the counts matrix."""
    adata = _get_adata(data)
    return float(adata.X.sum())


def filter_peaks(data: Union[AnnData, MuData], min_fragments_per_peak: int = 1,
                 non_overlapping: bool = True, delimiter: str = "-",
                 ix_return: bool = False):
    """Filter peaks by fragment count and, optionally, remove overlapping peaks.

    Faithful port of chromVAR's ``filterPeaks``: keeps peaks with at least
    ``min_fragments_per_peak`` fragments, then (if ``non_overlapping``) greedily
    removes overlapping peaks keeping the one with more fragments. Peaks must be
    sorted by genomic position for overlap removal.

    Parameters
    ----------
    data : AnnData or MuData
        Object with peak counts; peak names parsed as ``chrom<delim>start<delim>end``.
    min_fragments_per_peak : int, optional
        Minimum fragments to keep a peak, by default 1.
    non_overlapping : bool, optional
        Remove overlapping peaks, by default True.
    delimiter : str, optional
        Delimiter separating chrom/start/end in peak names, by default "-".
    ix_return : bool, optional
        Return kept peak indices instead of a filtered object, by default False.

    Returns
    -------
    np.ndarray of kept indices, or a filtered (copied) AnnData.
    """
    adata = _get_adata(data)
    fragments_per_peak = get_fragments_per_peak(adata)
    keep = np.flatnonzero(fragments_per_peak >= min_fragments_per_peak)

    if non_overlapping:
        chroms, starts, ends = [], [], []
        for name in adata.var_names:
            parts = re.split(delimiter, name)
            chroms.append(parts[0])
            starts.append(int(parts[1]))
            ends.append(int(parts[2]))
        chroms = np.asarray(chroms)
        starts = np.asarray(starts)
        ends = np.asarray(ends)

        def _is_disjoint(idx):
            if idx.size < 2:
                return True
            same = chroms[idx[:-1]] == chroms[idx[1:]]
            overlap = ends[idx[:-1]] >= starts[idx[1:]]
            return not np.any(same & overlap)

        # require sorted peaks (chromVAR raises otherwise)
        if not _is_disjoint(keep):
            order_ok = all(
                (chroms[keep[i]] != chroms[keep[i + 1]]) or
                (starts[keep[i]] <= starts[keep[i + 1]])
                for i in range(keep.size - 1))
            if not order_ok:
                raise ValueError(
                    "Peaks must be sorted to filter overlapping peaks; "
                    "please sort peaks by genomic position first.")

        while not _is_disjoint(keep):
            first = np.flatnonzero(
                (chroms[keep[:-1]] == chroms[keep[1:]]) &
                (ends[keep[:-1]] >= starts[keep[1:]]))
            second = first + 1
            # keep the peak with more fragments in each overlapping pair
            second_bigger = (fragments_per_peak[keep[second]] >
                             fragments_per_peak[keep[first]])
            discard = np.concatenate([keep[second[~second_bigger]],
                                      keep[first[second_bigger]]])
            keep = keep[~np.isin(keep, discard)]

    if ix_return:
        return keep
    return adata[:, keep].copy()


def filter_samples(data: Union[AnnData, MuData], min_depth: float = None,
                   min_in_peaks: float = None, depth: np.ndarray = None,
                   ix_return: bool = False):
    """Filter cells/samples by sequencing depth and fraction of reads in peaks.

    Faithful port of chromVAR's ``filterSamples``. ``depth`` (total reads per
    cell, including outside peaks) is taken from the ``depth`` argument, then
    ``.obs['depth']``; if neither is available it falls back to reads-in-peaks.
    Defaults match chromVAR: ``min_in_peaks`` = 0.5 x median(in-peak fraction),
    ``min_depth`` = max(500, 0.1 x median depth).

    Returns
    -------
    np.ndarray of kept indices, or a filtered (copied) AnnData.
    """
    adata = _get_adata(data)
    fragments_per_sample = get_fragments_per_sample(adata)

    if depth is None:
        if "depth" in adata.obs.columns:
            depth = adata.obs["depth"].values.astype(np.float64)
        else:
            depth = fragments_per_sample.astype(np.float64)
    depth = np.asarray(depth, dtype=np.float64)

    in_peaks = fragments_per_sample / depth
    if min_in_peaks is None:
        min_in_peaks = round(float(np.median(in_peaks)) * 0.5, 3)
    if min_depth is None:
        min_depth = max(500.0, float(np.median(depth)) * 0.1)

    keep = np.flatnonzero((depth >= min_depth) & (in_peaks >= min_in_peaks))

    if ix_return:
        return keep
    return adata[keep, :].copy()
