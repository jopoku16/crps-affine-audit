"""Non-weather affine validation on UCI bike-sharing counts."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "author_crps"))

from corrected_crps import (  # noqa: E402
    affine_standardized_weighted_compression,
    canonicalize_standardized,
    weighted_empirical_score,
)
from run_affine_extension import compressed_scores, fair_sorted  # noqa: E402
from utils.kernel_quadrature import kq_compress  # noqa: E402


DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DATA_DOI = "10.24432/C5W894"
ZIP_SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
OUTPUT = ROOT / "bike_sharing_affine_results.json"
ENSEMBLE_SIZE = 20
AUDIT_CASES = 2000
RANK = 10
NODES = 6
TOLERANCE = 2e-4


def fetch_hourly_data():
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != ZIP_SHA256:
        raise RuntimeError(f"dataset checksum {digest} did not match {ZIP_SHA256}")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        frame = pd.read_csv(archive.open("hour.csv"))
    return frame, digest


def build_past_only_ensembles(frame: pd.DataFrame):
    """Use the 20 most recent counts at the same hour and workday status."""
    ordered = frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["dteday"]) + pd.to_timedelta(
        ordered["hr"], unit="h"
    )
    ordered = ordered.sort_values("timestamp", kind="stable")
    histories = defaultdict(lambda: deque(maxlen=ENSEMBLE_SIZE))
    records = []
    for row in ordered.itertuples(index=False):
        key = (int(row.hr), int(row.workingday))
        history = histories[key]
        if len(history) == ENSEMBLE_SIZE:
            records.append(
                {
                    "instant": int(row.instant),
                    "timestamp": row.timestamp.isoformat(),
                    "observation": float(row.cnt),
                    "ensemble": np.asarray(history, dtype=np.float64).copy(),
                }
            )
        history.append(float(row.cnt))
    return records


def affine_score(samples, observation, seed):
    value, _, _, _, _ = affine_standardized_weighted_compression(
        samples, observation, RANK, NODES, seed
    )
    return float(value)


def legacy_bankers_coordinates(samples):
    x = torch.as_tensor(samples, dtype=torch.float64)
    scale = x.max() - x.min()
    return torch.round(((x - x.min()) / scale) * 1_000_000) / 1_000_000


def legacy_bankers_score(samples, observation, seed):
    x = torch.as_tensor(samples, dtype=torch.float64)
    y = torch.as_tensor(observation, dtype=torch.float64)
    location = x.min()
    scale = x.max() - location
    standardized = legacy_bankers_coordinates(x).float()
    standardized_observation = (y - location) / scale
    torch.manual_seed(int(seed))
    weights, nodes = kq_compress(standardized, RANK, NODES)
    return float(
        scale * weighted_empirical_score(weights, nodes, standardized_observation)
    )


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "rows": int(array.size),
        "median_absolute_change_rides": float(np.median(array)),
        "p95_absolute_change_rides": float(np.quantile(array, 0.95)),
        "maximum_absolute_change_rides": float(np.max(array)),
        "proportion_above_tolerance": float(np.mean(array > TOLERANCE)),
    }


def run_validation():
    frame, digest = fetch_hourly_data()
    eligible = build_past_only_ensembles(frame)
    selected = np.linspace(0, len(eligible) - 1, AUDIT_CASES, dtype=int)
    cases = [eligible[i] for i in selected]
    transformations = {
        "rides_to_hundreds": {"a": 0.01, "b": 0.0},
        "affine_stress_recoding": {"a": 0.01, "b": 10.0},
    }
    methods = (
        "released_compressed",
        "weighted_empirical",
        "affine_standardized",
        "full_fair_sorted",
    )
    changes = {
        transform: {method: [] for method in methods}
        for transform in transformations
    }
    failures = []
    legacy_half_grid_case_transform_mismatches = 0
    legacy_maximum_score_residual_rides = 0.0
    guarded_case_transform_mismatches = 0
    for position, case in enumerate(cases):
        samples = case["ensemble"]
        observation = case["observation"]
        seed = 20261100 + int(case["instant"])
        try:
            released, weighted = compressed_scores(
                samples, observation, RANK, NODES, seed
            )
            baseline = {
                "released_compressed": released,
                "weighted_empirical": weighted,
                "affine_standardized": affine_score(samples, observation, seed),
                "full_fair_sorted": fair_sorted(samples, observation),
            }
            for transform, pars in transformations.items():
                a, b = pars["a"], pars["b"]
                transformed_samples = a * samples + b
                transformed_observation = a * observation + b
                legacy_base = legacy_bankers_coordinates(samples)
                legacy_transformed = legacy_bankers_coordinates(
                    transformed_samples
                )
                if not torch.equal(legacy_base, legacy_transformed):
                    legacy_half_grid_case_transform_mismatches += 1
                    legacy_residual = abs(
                        legacy_bankers_score(
                            transformed_samples, transformed_observation, seed
                        )
                        / a
                        - legacy_bankers_score(samples, observation, seed)
                    )
                    legacy_maximum_score_residual_rides = max(
                        legacy_maximum_score_residual_rides, legacy_residual
                    )
                guarded_base = canonicalize_standardized(
                    (torch.as_tensor(samples, dtype=torch.float64) - samples.min())
                    / np.ptp(samples)
                )
                guarded_transformed = canonicalize_standardized(
                    (
                        torch.as_tensor(
                            transformed_samples, dtype=torch.float64
                        )
                        - transformed_samples.min()
                    )
                    / np.ptp(transformed_samples)
                )
                if not torch.equal(guarded_base, guarded_transformed):
                    guarded_case_transform_mismatches += 1
                released_t, weighted_t = compressed_scores(
                    transformed_samples,
                    transformed_observation,
                    RANK,
                    NODES,
                    seed,
                )
                transformed = {
                    "released_compressed": released_t,
                    "weighted_empirical": weighted_t,
                    "affine_standardized": affine_score(
                        transformed_samples, transformed_observation, seed
                    ),
                    "full_fair_sorted": fair_sorted(
                        transformed_samples, transformed_observation
                    ),
                }
                for method in methods:
                    changes[transform][method].append(
                        abs(transformed[method] / a - baseline[method])
                    )
        except Exception as exc:
            failures.append(
                {
                    "case_position": position,
                    "instant": case["instant"],
                    "error": repr(exc),
                }
            )
        if (position + 1) % 250 == 0:
            print(f"bike sharing: {position + 1}/{len(cases)}", flush=True)

    summary = {
        transform: {
            method: summarize(values) for method, values in method_values.items()
        }
        for transform, method_values in changes.items()
    }
    controls = {
        "failures": failures,
        "affine_standardized_maximum_rides": float(
            max(
                summary[t]["affine_standardized"]["maximum_absolute_change_rides"]
                for t in transformations
            )
        ),
        "full_fair_maximum_rides": float(
            max(
                summary[t]["full_fair_sorted"]["maximum_absolute_change_rides"]
                for t in transformations
            )
        ),
    }
    controls["pass"] = bool(
        not failures
        and controls["affine_standardized_maximum_rides"] <= TOLERANCE
        and controls["full_fair_maximum_rides"] <= TOLERANCE
    )

    return {
        "dataset": "UCI Bike Sharing hourly rental counts",
        "dataset_url": DATA_URL,
        "dataset_doi": DATA_DOI,
        "dataset_license": "CC BY 4.0",
        "zip_sha256": digest,
        "source_rows": int(len(frame)),
        "eligible_past_only_cases": int(len(eligible)),
        "audited_cases": int(len(cases)),
        "selection": "evenly spaced over all eligible cases",
        "forecast": {
            "description": "20 most recent earlier counts with the same hour and working-day status",
            "ensemble_members": ENSEMBLE_SIZE,
            "uses_future_data": False,
        },
        "configuration": [RANK, NODES],
        "tolerance_rides": TOLERANCE,
        "transformations": transformations,
        "summary": summary,
        "controls": controls,
        "development_diagnostic": {
            "status": "post-hoc software hardening check",
            "legacy_rule": "six-decimal torch.round (banker's rounding)",
            "legacy_half_grid_case_transform_mismatches": legacy_half_grid_case_transform_mismatches,
            "legacy_maximum_score_residual_rides": legacy_maximum_score_residual_rides,
            "guarded_half_up_case_transform_mismatches": guarded_case_transform_mismatches,
            "action": "retained the failure, added a regression test, and replaced the legacy rule with guarded half-up canonicalization",
        },
        "first_case": {
            "instant": cases[0]["instant"],
            "timestamp": cases[0]["timestamp"],
        },
        "last_case": {
            "instant": cases[-1]["instant"],
            "timestamp": cases[-1]["timestamp"],
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def main():
    result = run_validation()
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["controls"], indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
