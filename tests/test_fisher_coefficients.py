import importlib.util
import unittest


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


@unittest.skipUnless(_has_torch(), "torch is required for model_soup imports")
class TestFisherCoefficientHelpers(unittest.TestCase):
    def test_fisher_coefficients_are_simplex(self):
        from yolof.analysis.model_soup import fisher_branch_coefficients_from_traces

        coeffs = fisher_branch_coefficients_from_traces(
            cls_traces=[2.0, 3.0, 5.0],
            reg_traces=[1.0, 1.0, 2.0],
        )

        self.assertAlmostEqual(sum(coeffs["cls"]), 1.0)
        self.assertAlmostEqual(sum(coeffs["reg"]), 1.0)
        self.assertAlmostEqual(sum(coeffs["shared"]), 1.0)
        self.assertAlmostEqual(sum(coeffs["backbone_encoder"]), 1.0)

    def test_fisher_coefficients_validate_lengths(self):
        from yolof.analysis.model_soup import fisher_branch_coefficients_from_traces

        with self.assertRaises(ValueError):
            fisher_branch_coefficients_from_traces([1.0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
