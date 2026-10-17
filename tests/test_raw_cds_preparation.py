"""Scientific input-boundary tests; no alignment tools or fitting required."""
from pathlib import Path
import importlib.util
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("raw_cds_recipe", SOURCE/"scripts/prepare_cds_pairs.py")
recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe)


class RawCDS(unittest.TestCase):
    def test_translation_and_original_case(self):
        result = recipe.check_cds_pep("gene", "atgGCTTTTGGT", "MAFG")
        self.assertEqual(result["cds"], "atgGCTTTTGGT")

    def test_stop_requires_opt_in(self):
        with self.assertRaisesRegex(ValueError, "explicitly"):
            recipe.check_cds_pep("gene", "ATGGCTTAA", "MA")

    def test_only_terminal_stop_is_removed(self):
        r = recipe.check_cds_pep("gene", "ATGGCTtga", "MA*", True)
        self.assertEqual(r["cds"], "ATGGCT")
        self.assertEqual(r["removed_terminal_codon"], "tga")
        with self.assertRaisesRegex(ValueError, "internal stop"):
            recipe.check_cds_pep("bad", "ATGTGAGCTTAA", "MWA*", True)

    def test_terminal_marker_without_stop_is_not_repaired(self):
        with self.assertRaisesRegex(ValueError, "no terminal stop"):
            recipe.check_cds_pep("gene", "ATGGCT", "MA*", True)

    def test_frame_and_mismatch(self):
        for cds, pep in (("ATGG", "M"), ("ATGGCT", "MG"), ("ATGGCT", "MAA")):
            with self.subTest(cds=cds, pep=pep), self.assertRaises(ValueError):
                recipe.check_cds_pep("bad", cds, pep)

    def test_ambiguity_and_raw_gaps_are_reported(self):
        for cds, pep in (("ATGNNN", "MX"), ("ATG---", "M-"), ("ATGTGT", "MU")):
            with self.subTest(cds=cds, pep=pep), self.assertRaises(ValueError):
                recipe.check_cds_pep("bad", cds, pep)

    def test_projection_triplets_and_case(self):
        self.assertEqual(recipe.codon_projection("M-A-F", "atgGCTTTT", "MAF"), "atg---GCT---TTT")

    def test_residue_loss_cannot_pass(self):
        with self.assertRaisesRegex(ValueError, "lost"):
            recipe.codon_projection("MA", "ATGGCTTTT", "MAF")

    def test_identity_is_direction_sensitive(self):
        self.assertNotEqual(recipe.pair_identity("x", "ATG", "GCT"), recipe.pair_identity("x", "GCT", "ATG"))

    def test_fasta_duplicate_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"cds.fa"
            p.write_text(">a comment\nATG\n>a second\nATG\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                recipe.read_fasta(p)

    def test_pair_direction_and_literal_na_id(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"pairs.tsv"
            p.write_text("pair_id\tseq1\tseq2\nNA\ta\tb\nreverse\tb\ta\n")
            rows = recipe.read_pairs(p)
            self.assertEqual([r["pair_id"] for r in rows], ["NA", "reverse"])
            self.assertEqual(rows[1]["seq1"], "b")

    def test_duplicate_pair_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"pairs.tsv"
            p.write_text("pair_id\tseq1\tseq2\nx\ta\tb\nx\tb\ta\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                recipe.read_pairs(p)

    def test_missing_ids_fail_without_axt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/"cds.fa").write_text(">a\nATGGCT\n")
            (root/"pep.fa").write_text(">a\nMA\n")
            (root/"pairs.tsv").write_text("pair_id\tseq1\tseq2\nx\ta\tb\n")
            result = recipe.main(["--cds", str(root/"cds.fa"), "--pep", str(root/"pep.fa"),
                                  "--pairs", str(root/"pairs.tsv"), "--output", str(root/"out")])
            self.assertEqual(result, 2)
            self.assertFalse((root/"out/aligned_pairs.axt").exists())
            self.assertIn("missing", (root/"out/errors.tsv").read_text())

    def test_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d)/"valuable.txt"
            marker.write_text("original")
            result = recipe.main(["--cds", "missing", "--pep", "missing", "--pairs", "missing", "--output", d])
            self.assertEqual(result, 2)
            self.assertEqual(marker.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
