from typing import Union, Tuple
import re
import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import norm as _norm_dist
from anndata import AnnData
from mudata import MuData
from pysam import Fastafile
from tqdm import tqdm


def _bg_bin_structure(intensity: np.ndarray, bias: np.ndarray, w: float, bs: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic core of chromVAR ``getBackgroundPeaks``.

    Mirrors ``background_peaks.R::get_background_peaks_core`` (lines 127-149):
    whitens the (log10 reads, GC bias) space via the Cholesky factor of its
    covariance, lays a ``bs`` x ``bs`` grid over it, assigns each peak to its
    nearest grid bin and computes a Gaussian (sd = ``w``) distance kernel
    between bins.

    Returns
    -------
    bin_membership : (n_peaks,) int array, 0-based index of each peak's bin.
    bin_density : (bs*bs,) int array, number of peaks per bin.
    bin_p : (bs*bs, bs*bs) float array, ``dnorm(dist(bin_i, bin_j), 0, w)``.
    """
    norm_mat = np.column_stack([intensity, bias])  # (n, 2)

    # Whiten: norm_mat %*% inv(R) where R'R = cov(norm_mat). numpy's cholesky
    # returns the lower factor L (L L' = cov), and solve_triangular(L, X)
    # reproduces the R `forwardsolve(t(chol(cov)), .)` whitening exactly.
    chol_cov_mat = np.linalg.cholesky(np.cov(norm_mat, rowvar=False))
    trans_norm_mat = solve_triangular(
        a=chol_cov_mat, b=norm_mat.T, lower=True).T

    # Grid of bin centres; bins along axis 1 vary slowest (matches R rbind order)
    bins1 = np.linspace(trans_norm_mat[:, 0].min(), trans_norm_mat[:, 0].max(), bs)
    bins2 = np.linspace(trans_norm_mat[:, 1].min(), trans_norm_mat[:, 1].max(), bs)
    bin_data = np.column_stack([np.repeat(bins1, bs), np.tile(bins2, bs)])

    bin_p = _norm_dist.pdf(cdist(bin_data, bin_data), loc=0.0, scale=w)
    bin_membership = cKDTree(bin_data).query(trans_norm_mat, k=1)[1]
    bin_density = np.bincount(bin_membership, minlength=bs * bs)

    return bin_membership, bin_density, bin_p


def _sample_bg_peaks(bin_membership: np.ndarray, bin_density: np.ndarray,
                     bin_p: np.ndarray, niterations: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Sample background peaks (with replacement) per chromVAR ``bg_sample_helper``.

    Mirrors ``src/utils.cpp::bg_sample_helper`` (lines 83-99): for every peak,
    draw ``niterations`` background peaks from the whole peak set, weighting each
    candidate by ``dnorm(dist(its bin, this peak's bin), 0, w) / density(its bin)``.

    Because every peak in a bin shares that weight, this is equivalent to (and
    implemented as) sampling a candidate *bin* with probability proportional to
    the bin's distance kernel, then a uniform peak within it -- reducing the draw
    from O(n_peaks) candidates to O(n_occupied_bins). Returns a
    (n_peaks, niterations) array of 0-based peak indices.
    """
    n = bin_membership.shape[0]
    out = np.empty((n, niterations), dtype=np.int64)

    occ = np.flatnonzero(bin_density > 0)            # occupied bins
    # group peaks by bin: order[starts[t]:ends[t]] are the peaks in occ[t]
    order = np.argsort(bin_membership, kind="stable")
    sorted_bins = bin_membership[order]
    starts = np.searchsorted(sorted_bins, occ, side="left")
    ends = np.searchsorted(sorted_bins, occ, side="right")
    sizes = ends - starts

    for t, b in enumerate(occ):
        peaks_b = order[starts[t]:ends[t]]
        cnt = peaks_b.size
        # P(candidate bin j) proportional to kernel(j, b) over occupied bins
        p = bin_p[occ, b]
        p = p / p.sum()
        total = niterations * cnt
        chosen = rng.choice(occ.size, size=total, replace=True, p=p)
        # uniform peak within each chosen bin
        offs = (rng.random(total) * sizes[chosen]).astype(np.int64)
        out[peaks_b, :] = order[starts[chosen] + offs].reshape(cnt, niterations)

    return out


def get_bg_peaks(data: Union[AnnData, MuData], niterations: int = 50,
                 w: float = 0.1, bs: int = 50, seed: int = None, n_jobs=-1):
    """Find background peaks matched on GC bias and reads per peak.

    Faithful port of chromVAR's ``getBackgroundPeaks``: background peaks are
    *sampled* (with replacement) based on similarity in GC content and number of
    fragments using a Gaussian kernel over a binned, whitened feature space.

    Parameters
    ----------
    data : Union[AnnData, MuData]
        AnnData object with peak counts or MuData object with 'atac' modality
    niterations : int, optional
        Number of background peaks to sample per peak, by default 50
    w : float, optional
        Standard deviation of the Gaussian kernel controlling how similar
        background peaks must be, by default 0.1
    bs : int, optional
        Number of bins along each feature axis; higher is more precise but
        slower, by default 50
    seed : int, optional
        Seed for the random sampler, for reproducibility, by default None
    n_jobs : int, optional
        Retained for backwards compatibility; unused, by default -1

    Returns
    -------
    Updates `data` with ``varm['bg_peaks']``, a (n_peaks, niterations) array of
    background peak indices.
    """

    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError(
            "Expected AnnData or MuData object with 'atac' modality")

    # check if the object contains bias in Anndata.varm
    assert "gc_bias" in adata.var.columns, "Cannot find gc bias in the input object, please first run add_gc_bias!"

    fragments_per_peak = np.asarray(adata.X.sum(axis=0)).reshape(-1)
    if fragments_per_peak.min() <= 0:
        raise ValueError(
            "All peaks must have at least one fragment in one sample; "
            "please filter empty peaks before running get_bg_peaks.")

    intensity = np.log10(fragments_per_peak)
    bias = adata.var['gc_bias'].values.astype(np.float64)

    bin_membership, bin_density, bin_p = _bg_bin_structure(intensity, bias, w, bs)
    rng = np.random.default_rng(seed)
    adata.varm['bg_peaks'] = _sample_bg_peaks(
        bin_membership, bin_density, bin_p, niterations, rng)

    return None


def add_peak_seq(data: Union[AnnData, MuData], genome_file: str, delimiter="-"):
    """Add the DNA sequence of each peak to data object.

    Parameters
    ----------
    data : Union[AnnData, MuData]
        AnnData object with peak counts or MuData object with 'atac' modality.
    genome_file : str
        Filename of genome reference
    delimiter : str, optional
        Delimiter that separates peaks, by default "-"

    Returns
    -------
    Update `data`
    """

    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError(
            "Expected AnnData or MuData object with 'atac' modality")

    fasta = Fastafile(genome_file)
    adata.uns['peak_seq'] = [None] * adata.n_vars

    for i in tqdm(range(adata.n_vars)):
        peak = re.split(delimiter, adata.var_names[i])
        chrom, start, end = peak[0], int(peak[1]), int(peak[2])
        adata.uns['peak_seq'][i] = fasta.fetch(chrom, start, end).upper()

    return None


def add_gc_bias(data: Union[AnnData, MuData]):
    """Compute GC bias for each peak.

    Parameters
    ----------
    data : Union[AnnData, MuData]
        AnnData object with peak counts or MuData object with 'atac' modality.

    Returns
    -------
    Update data
    """

    if isinstance(data, AnnData):
        adata = data
    elif isinstance(data, MuData) and "atac" in data.mod:
        adata = data.mod["atac"]
    else:
        raise TypeError(
            "Expected AnnData or MuData object with 'atac' modality")

    assert "peak_seq" in adata.uns, \
        "Cannot find sequences, please first run add_peak_seq!"

    bias = np.zeros(adata.n_vars)

    for i in tqdm(range(adata.n_vars)):
        seq = adata.uns['peak_seq'][i]

        freq_a = seq.count("A")
        freq_c = seq.count("C")
        freq_g = seq.count("G")
        freq_t = seq.count("T")

        if freq_a + freq_c + freq_g + freq_t == 0:
            bias[i] = 0.5
        else:
            bias[i] = (freq_g + freq_c) / (freq_a + freq_c + freq_g + freq_t)

    adata.var['gc_bias'] = bias

    return None
