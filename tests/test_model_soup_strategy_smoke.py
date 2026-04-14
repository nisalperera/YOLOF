import unittest
import importlib.util


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


@unittest.skipUnless(_has_torch(), "torch is required for model_soup imports")
class TestModelSoupStrategySmoke(unittest.TestCase):
    def test_uniform_branch_coefficients_shape(self):
        from yolof.analysis.model_soup import uniform_branch_coefficients
        coeffs = uniform_branch_coefficients(6)

        for key in ["cls", "reg", "shared", "backbone_encoder"]:
            self.assertIn(key, coeffs)
            self.assertEqual(len(coeffs[key]), 6)
            self.assertAlmostEqual(sum(coeffs[key]), 1.0)

    def test_uniform_branch_coefficients_invalid(self):
        from yolof.analysis.model_soup import uniform_branch_coefficients
        with self.assertRaises(ValueError):
            uniform_branch_coefficients(0)


if __name__ == "__main__":
    unittest.main()
