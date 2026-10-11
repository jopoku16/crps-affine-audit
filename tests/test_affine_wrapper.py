import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REAL_DATA = ROOT / "real_data"
sys.path.insert(0, str(REAL_DATA / "author_crps"))
sys.path.insert(0, str(REAL_DATA))

from corrected_crps import (
    affine_standardized_weighted_compression,
    canonicalize_standardized,
)
from run_affine_extension import fair_pairwise, fair_sorted


class AffineWrapperTests(unittest.TestCase):
    def score(self, samples, observation, seed=27):
        value, _, _, _, _ = affine_standardized_weighted_compression(
            samples, observation, n_low_rank=5, n_compressed=3, seed=seed
        )
        return float(value)

    def test_positive_affine_equivariance(self):
        samples = np.array([-1.3, -0.2, 0.4, 1.1, 2.7, 4.0])
        observation = 0.65
        baseline = self.score(samples, observation)
        for scale, shift in [(2.5, 7.0), (1 / 25.4, 10.0), (0.2, -3.0)]:
            transformed = self.score(
                scale * samples + shift, scale * observation + shift
            )
            self.assertAlmostEqual(transformed, scale * baseline, places=11)

    def test_constant_ensemble_is_exact_point_mass(self):
        samples = np.repeat(4.25, 12)
        self.assertEqual(self.score(samples, 1.75), 2.5)
        self.assertEqual(self.score(samples, 4.25), 0.0)

    def test_sorted_and_pairwise_fair_crps_agree(self):
        samples = np.array([-1.2, -0.1, 0.0, 0.7, 2.3, 4.4])
        self.assertAlmostEqual(
            fair_sorted(samples, 0.4), fair_pairwise(samples, 0.4), places=14
        )

    def test_half_grid_boundary_is_canonical_after_recoding(self):
        samples = np.array(
            [
                34,
                68,
                69,
                72,
                7,
                82,
                58,
                93,
                49,
                51,
                36,
                77,
                45,
                75,
                95,
                102,
                14,
                83,
                135,
                61,
            ],
            dtype=float,
        )
        transformed = 0.01 * samples + 10.0
        base = canonicalize_standardized(
            (samples - samples.min()) / np.ptp(samples)
        ).numpy()
        recoded = canonicalize_standardized(
            (transformed - transformed.min()) / np.ptp(transformed)
        ).numpy()
        np.testing.assert_array_equal(base, recoded)
        score = self.score(samples, 177.0, seed=20263884)
        transformed_score = self.score(
            transformed, 0.01 * 177.0 + 10.0, seed=20263884
        )
        self.assertAlmostEqual(transformed_score / 0.01, score, places=10)


if __name__ == "__main__":
    unittest.main()
