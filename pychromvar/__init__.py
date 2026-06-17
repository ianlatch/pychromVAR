__version__ = "0.0.4"
__version_info__ = tuple([int(num) for num in __version__.split('.')])  # noqa: F401

from .preprocessing import get_bg_peaks, add_gc_bias, add_peak_seq
from .match_motif import match_motif
from .compute_deviations import compute_deviations, compute_expectation
from .compute_variability import compute_variability
from .differential import differential_deviations, differential_variability
from .filtering import (get_fragments_per_peak, get_fragments_per_sample,
                        get_total_fragments, filter_peaks, filter_samples)
from .get_genome import get_genome

