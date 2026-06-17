__version__ = "0.1.0"
__version_info__ = tuple([int(num) for num in __version__.split('.')])  # noqa: F401

from .preprocessing import get_bg_peaks, add_gc_bias, add_peak_seq
from .match_motif import match_motif
from .compute_deviations import compute_deviations, compute_expectation
from .compute_variability import compute_variability
from .differential import differential_deviations, differential_variability
from .filtering import (get_fragments_per_peak, get_fragments_per_sample,
                        get_total_fragments, filter_peaks, filter_samples)
from .get_genome import get_genome

__all__ = [
    "get_bg_peaks", "add_gc_bias", "add_peak_seq",
    "match_motif",
    "compute_deviations", "compute_expectation",
    "compute_variability",
    "differential_deviations", "differential_variability",
    "get_fragments_per_peak", "get_fragments_per_sample",
    "get_total_fragments", "filter_peaks", "filter_samples",
    "get_genome",
]

