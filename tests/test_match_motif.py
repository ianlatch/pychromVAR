import numpy as np
import pysam

from pychromvar.match_motif import _bg_from_genome


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
