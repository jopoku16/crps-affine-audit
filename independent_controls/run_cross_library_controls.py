"""Run affine metamorphic tests across public CRPS package interfaces.

This experiment is a negative control for the source-specific audit. It asks
whether standard empirical and weighted empirical CRPS implementations obey
the expected positive-affine transformation law. It does not test compression.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import properscoring
import scoringrules
import xarray as xr
import xskillscore
from scores.probability import crps_for_ensemble


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "cross_library_results.json"
SEED = 20261011
CASES_PER_CELL = 250
ENSEMBLE_SIZES = (5, 11, 50, 101)
TRANSFORMS = (
    (1.0, 273.15, "origin_plus_273_15"),
    (1.0, -10_000.0, "origin_minus_10000"),
    (1.0 / 25.4, 10.0, "divide_25_4_plus_10"),
    (25.4, -32.0, "times_25_4_minus_32"),
    (1_000.0, 100_000.0, "times_1000_plus_100000"),
)
ABSOLUTE_TOLERANCE = 1e-8


def sorted_empirical_crps(forecasts: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Exact empirical CRPS in O(M log M) time for each row."""
    x = np.sort(np.asarray(forecasts, dtype=np.float64), axis=-1)
    y = np.asarray(observations, dtype=np.float64)
    m = x.shape[-1]
    first = np.mean(np.abs(x - y[:, None]), axis=-1)
    coefficients = 2 * np.arange(1, m + 1, dtype=np.float64) - m - 1
    spread = np.sum(x * coefficients, axis=-1) / (m * m)
    return first - spread


def weighted_empirical_crps(
    forecasts: np.ndarray, observations: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Direct weighted empirical CRPS, used only as an audit reference."""
    x = np.asarray(forecasts, dtype=np.float64)
    y = np.asarray(observations, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w, axis=-1, keepdims=True)
    first = np.sum(w * np.abs(x - y[:, None]), axis=-1)
    distances = np.abs(x[:, :, None] - x[:, None, :])
    spread = 0.5 * np.einsum("bi,bij,bj->b", w, distances, w)
    return first - spread


def package_versions() -> dict[str, str]:
    names = ("numpy", "properscoring", "scoringrules", "scores", "xarray", "xskillscore")
    return {name: importlib.metadata.version(name) for name in names}


def make_cases(rng: np.random.Generator, family: str, count: int, m: int):
    if family == "normal":
        x = rng.normal(size=(count, m))
        y = rng.normal(size=count)
    elif family == "lognormal":
        x = rng.lognormal(mean=0.0, sigma=0.8, size=(count, m))
        y = rng.lognormal(mean=0.0, sigma=0.8, size=count)
    elif family == "student_t_3":
        x = rng.standard_t(df=3, size=(count, m)) / np.sqrt(3.0)
        y = rng.standard_t(df=3, size=count) / np.sqrt(3.0)
    elif family == "two_component_mixture":
        selectors = rng.random((count, m)) < 0.7
        x = np.where(
            selectors,
            rng.normal(-1.0, 0.5, size=(count, m)),
            rng.normal(2.0, 1.2, size=(count, m)),
        )
        y_selectors = rng.random(count) < 0.7
        y = np.where(
            y_selectors,
            rng.normal(-1.0, 0.5, size=count),
            rng.normal(2.0, 1.2, size=count),
        )
    else:
        raise ValueError(f"unknown family: {family}")
    return x.astype(np.float64), y.astype(np.float64)


def equal_weight_scores(forecasts: np.ndarray, observations: np.ndarray):
    fcst = xr.DataArray(forecasts, dims=("case", "member"))
    obs = xr.DataArray(observations, dims=("case",))
    return {
        "local_sorted_reference": sorted_empirical_crps(forecasts, observations),
        "properscoring": np.asarray(
            properscoring.crps_ensemble(observations, forecasts), dtype=np.float64
        ),
        "scoringrules": np.asarray(
            scoringrules.crps_ensemble(
                observations, forecasts, estimator="nrg", backend="numpy"
            ),
            dtype=np.float64,
        ),
        "scores": np.asarray(
            crps_for_ensemble(
                fcst,
                obs,
                "member",
                method="ecdf",
                preserve_dims=["case"],
            ).values,
            dtype=np.float64,
        ),
        "xskillscore": np.asarray(
            xskillscore.crps_ensemble(
                obs, fcst, member_dim="member", dim=[]
            ).values,
            dtype=np.float64,
        ),
    }


def weighted_scores(
    forecasts: np.ndarray, observations: np.ndarray, weights: np.ndarray
):
    return {
        "local_weighted_reference": weighted_empirical_crps(
            forecasts, observations, weights
        ),
        "properscoring_weighted": np.asarray(
            properscoring.crps_ensemble(
                observations, forecasts, weights=weights
            ),
            dtype=np.float64,
        ),
        "scoringrules_weighted": np.asarray(
            scoringrules.crps_ensemble(
                observations,
                forecasts,
                ens_w=weights,
                estimator="nrg",
                backend="numpy",
            ),
            dtype=np.float64,
        ),
    }


def empty_summary():
    return {
        "comparisons": 0,
        "failures": 0,
        "maximum_mapped_back_error": 0.0,
        "maximum_reference_difference": 0.0,
    }


def update_summary(summary, baseline, transformed, scale, reference):
    mapped = np.asarray(transformed, dtype=np.float64) / scale
    error = np.abs(mapped - baseline)
    summary["comparisons"] += int(error.size)
    summary["failures"] += int(np.count_nonzero(error > ABSOLUTE_TOLERANCE))
    summary["maximum_mapped_back_error"] = max(
        summary["maximum_mapped_back_error"], float(np.max(error))
    )
    summary["maximum_reference_difference"] = max(
        summary["maximum_reference_difference"],
        float(np.max(np.abs(baseline - reference))),
    )


def run_experiment():
    rng = np.random.default_rng(SEED)
    families = ("normal", "lognormal", "student_t_3", "two_component_mixture")
    equal_summary = {
        name: empty_summary()
        for name in (
            "local_sorted_reference",
            "properscoring",
            "scoringrules",
            "scores",
            "xskillscore",
        )
    }
    weighted_summary = {
        name: empty_summary()
        for name in (
            "local_weighted_reference",
            "properscoring_weighted",
            "scoringrules_weighted",
        )
    }

    for family in families:
        for m in ENSEMBLE_SIZES:
            forecasts, observations = make_cases(rng, family, CASES_PER_CELL, m)
            weights = rng.dirichlet(np.ones(m), size=CASES_PER_CELL)
            equal_base = equal_weight_scores(forecasts, observations)
            weighted_base = weighted_scores(forecasts, observations, weights)
            equal_reference = equal_base["local_sorted_reference"]
            weighted_reference = weighted_base["local_weighted_reference"]

            for scale, shift, _ in TRANSFORMS:
                transformed_forecasts = scale * forecasts + shift
                transformed_observations = scale * observations + shift
                equal_transformed = equal_weight_scores(
                    transformed_forecasts, transformed_observations
                )
                weighted_transformed = weighted_scores(
                    transformed_forecasts, transformed_observations, weights
                )
                for name, baseline in equal_base.items():
                    update_summary(
                        equal_summary[name],
                        baseline,
                        equal_transformed[name],
                        scale,
                        equal_reference,
                    )
                for name, baseline in weighted_base.items():
                    update_summary(
                        weighted_summary[name],
                        baseline,
                        weighted_transformed[name],
                        scale,
                        weighted_reference,
                    )

    cases = len(families) * len(ENSEMBLE_SIZES) * CASES_PER_CELL
    return {
        "study": "cross-library affine metamorphic controls",
        "seed": SEED,
        "families": list(families),
        "ensemble_sizes": list(ENSEMBLE_SIZES),
        "cases_per_family_size_cell": CASES_PER_CELL,
        "total_forecast_cases": cases,
        "transformations_per_case": len(TRANSFORMS),
        "transformations": [
            {"scale": scale, "shift": shift, "label": label}
            for scale, shift, label in TRANSFORMS
        ],
        "absolute_tolerance_in_original_units": ABSOLUTE_TOLERANCE,
        "package_versions": package_versions(),
        "implementation_notes": {
            "properscoring": "public Python project",
            "scoringrules": "separate public Python project",
            "scores": "separate public project; its documentation records adapted CRPS code",
            "xskillscore": "xarray interface that calls properscoring for ensemble CRPS",
        },
        "equal_weight": equal_summary,
        "unequal_weight": weighted_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_experiment()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
