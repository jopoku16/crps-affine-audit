"""Affine-recoding and runtime extension for the CRPS implementation audit."""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "author_crps"))

from corrected_crps import (  # noqa: E402
    affine_standardized_weighted_compression,
    anchored_weighted_compression,
)
from utils.kernel_quadrature import indicator_kernel, kq_compress  # noqa: E402

torch.set_num_threads(1)


def weighted_empirical_from_nodes(weights, nodes, observation):
    w = weights.double()
    z = nodes.double()
    w = w / w.sum()
    distances = (z[:, None] - z[None, :]).abs()
    return float(w @ (z - observation).abs() - 0.5 * (w @ distances @ w))


def compressed_scores(samples, observation, rank, nodes, seed):
    x = torch.as_tensor(samples, dtype=torch.float32)
    y = float(np.float32(observation))
    torch.manual_seed(int(seed))
    weights, selected = kq_compress(x, rank, nodes)
    wd = weights.double()
    zd = selected.double()
    normalized = wd / wd.sum()
    kernel = indicator_kernel(zd, zd)
    released = (
        wd @ (zd - y).abs()
        + wd @ zd
        - 2 * nodes / (nodes - 1) * (wd @ kernel @ wd)
    )
    weighted = weighted_empirical_from_nodes(normalized, zd, y)
    return float(released), weighted


def fair_sorted(samples, observation):
    x = np.sort(np.asarray(samples, dtype=np.float64))
    m = len(x)
    if m == 1:
        return float(abs(x[0] - observation))
    coefficients = 2 * np.arange(1, m + 1) - m - 1
    spread = float(coefficients @ x) / (m * (m - 1))
    return float(np.mean(np.abs(x - observation)) - spread)


def fair_pairwise(samples, observation):
    x = np.asarray(samples, dtype=np.float64)
    m = len(x)
    if m == 1:
        return float(abs(x[0] - observation))
    spread = np.abs(x[:, None] - x[None, :]).sum() / (2 * m * (m - 1))
    return float(np.mean(np.abs(x - observation)) - spread)


def anchored_score(samples, observation, rank, nodes, seed):
    value, _, _ = anchored_weighted_compression(
        samples, observation, rank, nodes, seed
    )
    return float(value)


def affine_score(samples, observation, rank, nodes, seed):
    value, _, _, _, _ = affine_standardized_weighted_compression(
        samples, observation, rank, nodes, seed
    )
    return float(value)


def summarize_changes(changes):
    z = np.asarray(changes, dtype=np.float64)
    return {
        "rows": int(len(z)),
        "median_absolute_change_mm": float(np.median(z)),
        "p95_absolute_change_mm": float(np.quantile(z, 0.95)),
        "maximum_absolute_change_mm": float(np.max(z)),
        "proportion_above_2e_4": float(np.mean(z > 2e-4)),
    }


def precipitation_affine_validation():
    data = pd.read_csv(ROOT / "rain.csv")
    x = data.iloc[:, 1:12].to_numpy(float)
    y = data.iloc[:, 0].to_numpy(float)
    rank, nodes = 8, 5
    transformations = {
        "millimetres_to_inches": {"a": 1 / 25.4, "b": 0.0},
        "positive_affine_recoding": {"a": 1 / 25.4, "b": 10.0},
    }
    methods = (
        "released",
        "weighted_empirical",
        "translation_anchored",
        "affine_standardized",
        "full_fair_sorted",
    )
    changes = {
        transform: {method: [] for method in methods}
        for transform in transformations
    }
    failures = []
    started = time.perf_counter()

    for i in range(len(y)):
        seed = 20261010 + i
        try:
            released, weighted = compressed_scores(x[i], y[i], rank, nodes, seed)
            anchored = anchored_score(x[i], y[i], rank, nodes, seed)
            affine = affine_score(x[i], y[i], rank, nodes, seed)
            fair = fair_sorted(x[i], y[i])
            baseline = {
                "released": released,
                "weighted_empirical": weighted,
                "translation_anchored": anchored,
                "affine_standardized": affine,
                "full_fair_sorted": fair,
            }
            for transform, pars in transformations.items():
                a, b = pars["a"], pars["b"]
                xt = a * x[i] + b
                yt = a * y[i] + b
                released_t, weighted_t = compressed_scores(
                    xt, yt, rank, nodes, seed
                )
                transformed = {
                    "released": released_t,
                    "weighted_empirical": weighted_t,
                    "translation_anchored": anchored_score(
                        xt, yt, rank, nodes, seed
                    ),
                    "affine_standardized": affine_score(
                        xt, yt, rank, nodes, seed
                    ),
                    "full_fair_sorted": fair_sorted(xt, yt),
                }
                for method in methods:
                    changes[transform][method].append(
                        abs(transformed[method] / a - baseline[method])
                    )
        except Exception as exc:
            failures.append({"row": i, "error": repr(exc)})
        if (i + 1) % 500 == 0:
            print(f"precipitation: {i + 1}/{len(y)}", flush=True)

    summary = {
        transform: {
            method: summarize_changes(values)
            for method, values in method_changes.items()
        }
        for transform, method_changes in changes.items()
    }
    controls = {
        "failures": failures,
        "constant_ensemble_rows": int(
            np.sum(np.ptp(x, axis=1) == 0)
        ),
        "affine_standardized_maximum_mm": float(
            max(
                summary[t]["affine_standardized"]["maximum_absolute_change_mm"]
                for t in transformations
            )
        ),
        "full_fair_maximum_mm": float(
            max(
                summary[t]["full_fair_sorted"]["maximum_absolute_change_mm"]
                for t in transformations
            )
        ),
    }
    controls["pass"] = bool(
        not failures
        and controls["affine_standardized_maximum_mm"] <= 2e-4
        and controls["full_fair_maximum_mm"] <= 2e-4
    )

    rows = []
    for transform, method_changes in summary.items():
        for method, values in method_changes.items():
            rows.append({"transformation": transform, "method": method, **values})
    pd.DataFrame(rows).to_csv(
        ROOT / "precipitation_affine_summary.csv", index=False
    )
    return {
        "dataset": "Innsbruck 12-hour precipitation ensemble",
        "rows": int(len(y)),
        "ensemble_members": int(x.shape[1]),
        "configuration": [rank, nodes],
        "transformations": transformations,
        "summary": summary,
        "controls": controls,
        "runtime_seconds": time.perf_counter() - started,
    }


def normal_population_crps(observation):
    z = float(observation)
    return float(z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))


def runtime_accuracy_benchmark():
    rng = np.random.default_rng(20261010)
    sizes = (100, 500, 2000)
    repetitions = 12
    rank, nodes = 40, 20
    observation = 0.25
    truth = normal_population_crps(observation)
    methods = (
        "released_compressed",
        "affine_standardized",
        "full_fair_sorted",
        "full_fair_pairwise",
    )
    rows = []

    # Warm up imports, Torch kernels, and the compressor outside timed runs.
    warm = rng.normal(size=100)
    compressed_scores(warm, observation, rank, nodes, 1)
    affine_score(warm, observation, rank, nodes, 1)

    for size in sizes:
        records = {method: {"times": [], "errors": []} for method in methods}
        for rep in range(repetitions):
            samples = rng.normal(size=size)
            seed = 20265000 + size * 100 + rep

            start = time.perf_counter()
            released, _ = compressed_scores(
                samples, observation, rank, nodes, seed
            )
            records["released_compressed"]["times"].append(
                time.perf_counter() - start
            )
            records["released_compressed"]["errors"].append(released - truth)

            start = time.perf_counter()
            affine = affine_score(samples, observation, rank, nodes, seed)
            records["affine_standardized"]["times"].append(
                time.perf_counter() - start
            )
            records["affine_standardized"]["errors"].append(affine - truth)

            start = time.perf_counter()
            sorted_value = fair_sorted(samples, observation)
            records["full_fair_sorted"]["times"].append(
                time.perf_counter() - start
            )
            records["full_fair_sorted"]["errors"].append(sorted_value - truth)

            start = time.perf_counter()
            pairwise_value = fair_pairwise(samples, observation)
            records["full_fair_pairwise"]["times"].append(
                time.perf_counter() - start
            )
            records["full_fair_pairwise"]["errors"].append(
                pairwise_value - truth
            )

        for method, values in records.items():
            times = np.asarray(values["times"])
            errors = np.asarray(values["errors"])
            rows.append(
                {
                    "ensemble_size": size,
                    "method": method,
                    "repetitions": repetitions,
                    "median_milliseconds": float(1000 * np.median(times)),
                    "p95_milliseconds": float(1000 * np.quantile(times, 0.95)),
                    "rmse_vs_population_crps": float(
                        np.sqrt(np.mean(errors**2))
                    ),
                    "mean_error": float(np.mean(errors)),
                }
            )
        print(f"runtime benchmark: M={size}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "runtime_accuracy_summary.csv", index=False)
    return {
        "distribution": "standard Normal",
        "observation": observation,
        "population_crps": truth,
        "configuration": [rank, nodes],
        "rows": rows,
    }


def main():
    precipitation = precipitation_affine_validation()
    runtime = runtime_accuracy_benchmark()
    output = {
        "method_name": "affine-standardized weighted CRPS wrapper",
        "source_commit": "dddd3ea46a6199d2ed48019c7825e79fb242787e",
        "precipitation": precipitation,
        "runtime_accuracy": runtime,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    path = ROOT / "affine_extension_results.json"
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps(precipitation["controls"], indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
