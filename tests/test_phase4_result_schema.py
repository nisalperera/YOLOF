import unittest


class TestPhase4ResultSchema(unittest.TestCase):
    def test_branch_barrier_schema_keys(self):
        sample = {
            "decoder_barriers_branch": {
                "cls": [],
                "reg": [],
                "shared": [],
                "full_decoder": [],
            },
            "branch_comparisons": {
                "cls_vs_reg": {
                    "lhs": "cls",
                    "rhs": "reg",
                    "count": 0,
                    "mean_delta": 0.0,
                    "median_delta": 0.0,
                    "std_delta": 0.0,
                    "min_delta": 0.0,
                    "max_delta": 0.0,
                    "positive_fraction": 0.0,
                    "wilcoxon": {},
                    "per_pair": [],
                },
            },
            "branch_alpha_steps": 21,
        }

        self.assertIn("decoder_barriers_branch", sample)
        self.assertIn("branch_comparisons", sample)
        self.assertIn("branch_alpha_steps", sample)
        self.assertEqual(sample["branch_alpha_steps"], 21)

        branch = sample["decoder_barriers_branch"]
        for key in ["cls", "reg", "shared", "full_decoder"]:
            self.assertIn(key, branch)

        cmp_item = sample["branch_comparisons"]["cls_vs_reg"]
        for key in [
            "lhs",
            "rhs",
            "count",
            "mean_delta",
            "median_delta",
            "std_delta",
            "min_delta",
            "max_delta",
            "positive_fraction",
            "wilcoxon",
            "per_pair",
        ]:
            self.assertIn(key, cmp_item)


if __name__ == "__main__":
    unittest.main()
