import unittest
from pathlib import Path


class PartialSubsetRoutingTests(unittest.TestCase):

    def test_partial_feature_selection_does_not_destroy_full_registry(self):
        src = (
            Path(__file__).resolve().parents[1]
            / "hcc_postoperative_prognosis_benchmark.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn(
            "feature_sets = {name: feature_sets[name] for name in args.feature_sets}",
            src,
        )

        self.assertIn(
            "selected_feature_sets = list(args.feature_sets)",
            src,
        )

        self.assertIn(
            "if fs in cfg.selected_feature_sets",
            src,
        )

        self.assertIn(
            'cfg.feature_sets["classic_preop"][-1]',
            src,
        )


if __name__ == "__main__":
    unittest.main()
