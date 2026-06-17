import numpy as np
import pysam
from anndata import AnnData

from pychromvar.match_motif import _bg_from_genome, match_motif


class _Motif:
    """Minimal stand-in for a Bio.motifs / pyjaspar motif."""
    def __init__(self, matrix_id, name, consensus):
        self.matrix_id = matrix_id
        self.name = name
        self.counts = {b: [100.0 if base == b else 1.0 for base in consensus]
                       for b in "ACGT"}


def test_bg_from_genome_frequencies(tmp_path):
    # contig1: 4 A, 2 C, 2 G, 2 T ; contig2: 2 A, 2 C, 4 G, 2 T
    # totals -> A=6, C=4, G=6, T=4 over 20 bases
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\nAAAACCGGTT\n>chr2\nAACCGGGGTT\n")
    pysam.faidx(str(fasta))

    bg = _bg_from_genome(str(fasta))

    assert len(bg) == 4
    assert np.isclose(sum(bg), 1.0)
    # order is A, C, G, T
    assert np.allclose(bg, [6 / 20, 4 / 20, 6 / 20, 4 / 20])


def test_match_motif_extraction():
    motifs = [_Motif("M1", "ACGT", "ACGTACGT"),
              _Motif("M2", "polyT", "TTTTTTTT")]
    data = AnnData(np.ones((2, 2), dtype=np.float32))
    data.uns["peak_seq"] = ["ACGTACGTACGTACGTACGT", "TTTTTTTTTTTTTTTTTTTT"]

    match_motif(data, motifs, background="even", p_value=0.05)

    mm = data.varm["motif_match"]
    assert mm.shape == (2, 2)
    assert mm.dtype == np.uint8
    assert set(np.unique(mm)).issubset({0, 1})
    # peak 0 carries the ACGT motif; peak 1 carries the poly-T motif
    assert mm[0, 0] == 1 and mm[0, 1] == 0
    assert mm[1, 1] == 1 and mm[1, 0] == 0
    assert data.uns["motif_name"] == ["M1.ACGT", "M2.polyT"]
