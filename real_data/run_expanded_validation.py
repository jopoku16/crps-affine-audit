import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "author_crps"))
from utils.kernel_quadrature import indicator_kernel, kq_compress  # noqa: E402
from corrected_crps import anchored_weighted_compression  # noqa: E402

torch.set_num_threads(1)


def fair_score(x, y):
    x = np.asarray(x, dtype=np.float64)
    m = len(x)
    return float(np.mean(np.abs(x - y)) - np.abs(x[:, None] - x[None, :]).sum() / (2 * m * (m - 1)))


def empirical_score(x, y):
    x = np.asarray(x, dtype=np.float64)
    m = len(x)
    return float(np.mean(np.abs(x - y)) - np.abs(x[:, None] - x[None, :]).sum() / (2 * m * m))


def raw_scores(x, y, rank, nodes, seed):
    tx = torch.tensor(x, dtype=torch.float32)
    ty = float(np.float32(y))
    torch.manual_seed(int(seed))
    w, z = kq_compress(tx, rank, nodes)
    wd, zd = w.double(), z.double()
    n = wd / wd.sum()
    h = indicator_kernel(zd, zd)
    d = (zd[:, None] - zd[None, :]).abs()
    released = wd @ (zd - ty).abs() + wd @ zd - 2 * nodes / (nodes - 1) * (wd @ h @ wd)
    q = n @ n
    normalized = n @ (zd - ty).abs() + n @ zd - 2 / (1 - q) * (n @ h @ n)
    weighted = n @ (zd - ty).abs() - 0.5 * (n @ d @ n)
    return np.array([released.item(), normalized.item(), weighted.item()]), float(abs(wd.sum() - 1)), float(w.min())


def anchored_score(x, y, rank, nodes, seed):
    value, w, _ = anchored_weighted_compression(x, y, rank, nodes, seed)
    wd = w.double()
    return float(value), float(abs(wd.sum() - 1)), float(w.min())


def moving_block_indices(n, block, rng):
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]


def bootstrap_change_summary(changes, kind, groups=None):
    out = {}
    if kind == "moving":
        for block in (7, 14, 30):
            rng = np.random.default_rng(20261009 + block)
            stats = np.empty((2000, changes.shape[1], 2))
            for b in range(2000):
                idx = moving_block_indices(len(changes), block, rng)
                sample = changes[idx]
                stats[b, :, 0] = np.median(sample, axis=0)
                stats[b, :, 1] = np.mean(sample > 0.01, axis=0)
            out[f"moving_block_{block}"] = {
                "median_CI95": np.quantile(stats[:, :, 0], [0.025, 0.975], axis=0).T.tolist(),
                "proportion_above_0_01_CI95": np.quantile(stats[:, :, 1], [0.025, 0.975], axis=0).T.tolist(),
            }
    else:
        labels = np.asarray(groups)
        unique = np.unique(labels)
        by_group = [np.flatnonzero(labels == value) for value in unique]
        rng = np.random.default_rng(20261009)
        stats = np.empty((2000, changes.shape[1], 2))
        for b in range(2000):
            chosen = rng.integers(0, len(unique), size=len(unique))
            idx = np.concatenate([by_group[j] for j in chosen])
            sample = changes[idx]
            stats[b, :, 0] = np.median(sample, axis=0)
            stats[b, :, 1] = np.mean(sample > 0.01, axis=0)
        out["date_cluster"] = {
            "median_CI95": np.quantile(stats[:, :, 0], [0.025, 0.975], axis=0).T.tolist(),
            "proportion_above_0_01_CI95": np.quantile(stats[:, :, 1], [0.025, 0.975], axis=0).T.tolist(),
        }
    return out


def load_data():
    inns = pd.read_csv(ROOT / "temp.csv")
    srft = pd.read_csv(ROOT / "srft.csv")
    return {
        "Innsbruck": {
            "x": inns.iloc[:, 1:12].to_numpy(float),
            "y": inns.iloc[:, 0].to_numpy(float),
            "shifts": (0.0, 273.15),
            "labels": ("Celsius", "Kelvin"),
            "config": (8, 5),
            "configs": [(5, 3), (8, 3), (8, 5), (10, 5), (10, 8)],
            "groups": None,
        },
        "Pacific Northwest": {
            "x": srft.iloc[:, 0:8].to_numpy(float),
            "y": srft["observation"].to_numpy(float),
            "shifts": (0.0, -273.15),
            "labels": ("Kelvin", "Celsius"),
            "config": (6, 4),
            "configs": [(4, 3), (6, 3), (6, 4), (7, 4), (7, 6)],
            "groups": srft["date"].astype(str).to_numpy(),
        },
    }


def full_validation(name, data):
    x, y = data["x"], data["y"]
    rank, nodes = data["config"]
    values = np.full((len(y), 2, 5), np.nan)
    failures = []
    max_mass, min_weight = 0.0, np.inf
    started = time.perf_counter()
    for i in range(len(y)):
        for oi, shift in enumerate(data["shifts"]):
            try:
                raw, mass, low = raw_scores(x[i] + shift, y[i] + shift, rank, nodes, 20261008 + i)
                anchored, amass, alow = anchored_score(x[i] + shift, y[i] + shift, rank, nodes, 20261008 + i)
                values[i, oi, :3] = raw
                values[i, oi, 3] = anchored
                values[i, oi, 4] = fair_score(x[i] + shift, y[i] + shift)
                max_mass = max(max_mass, mass, amass)
                min_weight = min(min_weight, low, alow)
            except Exception as exc:
                failures.append({"row": i, "origin": oi, "error": repr(exc)})
        if (i + 1) % 1000 == 0:
            print(f"{name}: {i + 1}/{len(y)}", flush=True)
    changes = np.abs(values[:, 1] - values[:, 0])
    methods = ["released", "normalized_pair", "weighted_empirical", "anchored_weighted", "full_fair"]
    summary = {}
    for j, method in enumerate(methods):
        valid = np.isfinite(changes[:, j])
        z = changes[valid, j]
        summary[method] = {
            "valid_rows": int(valid.sum()),
            "median": float(np.median(z)),
            "p95": float(np.quantile(z, 0.95)),
            "maximum": float(np.max(z)),
            "proportion_above_0_01": float(np.mean(z > 0.01)),
        }
    kind = "moving" if data["groups"] is None else "cluster"
    uncertainty = bootstrap_change_summary(changes, kind, data["groups"])
    controls = {
        "failures": failures,
        "maximum_weight_mass_error": max_mass,
        "minimum_weight": min_weight,
        "maximum_full_fair_change": float(np.nanmax(changes[:, 4])),
        "maximum_anchored_change": float(np.nanmax(changes[:, 3])),
    }
    controls["pass"] = bool(not failures and max_mass <= 2e-4 and min_weight >= 0 and controls["maximum_full_fair_change"] <= 1e-10 and controls["maximum_anchored_change"] <= 2e-4)
    return {"rows": len(y), "configuration": [rank, nodes], "controls": controls, "summary": summary, "uncertainty": uncertainty, "runtime_seconds": time.perf_counter() - started}, values


def sensitivity(name, data):
    x, y = data["x"], data["y"]
    indices = np.unique(np.linspace(0, len(y) - 1, 400).round().astype(int))
    result = {}
    for rank, nodes in data["configs"]:
        changes = []
        failures = []
        for ii, i in enumerate(indices):
            for rep in range(10):
                seed = 20262000 + rep * 100000 + int(i)
                try:
                    pair = []
                    for shift in data["shifts"]:
                        raw, _, _ = raw_scores(x[i] + shift, y[i] + shift, rank, nodes, seed)
                        anchored, _, _ = anchored_score(x[i] + shift, y[i] + shift, rank, nodes, seed)
                        pair.append([raw[0], raw[2], anchored])
                    changes.append(np.abs(np.asarray(pair[1]) - np.asarray(pair[0])))
                except Exception as exc:
                    failures.append({"row": int(i), "seed": seed, "error": repr(exc)})
            if (ii + 1) % 100 == 0:
                print(f"{name} sensitivity {rank}/{nodes}: {ii + 1}/{len(indices)}", flush=True)
        arr = np.asarray(changes)
        result[f"{rank}_{nodes}"] = {
            "row_seed_pairs": len(arr),
            "failures": failures,
            "methods": {
                method: {"median": float(np.median(arr[:, j])), "p95": float(np.quantile(arr[:, j], 0.95)), "maximum": float(np.max(arr[:, j]))}
                for j, method in enumerate(("released", "weighted_empirical", "anchored_weighted"))
            },
        }
    return result


def equal_budget(name, data):
    x, y = data["x"], data["y"]
    rank, nodes = data["config"]
    indices = np.unique(np.linspace(0, len(y) - 1, 400).round().astype(int))
    methods = ["released", "weighted_empirical", "anchored_weighted", "random_subsample_fair", "quantile_empirical"]
    errors = {method: [] for method in methods}
    elapsed = {method: 0.0 for method in methods}
    for i in indices:
        ref = fair_score(x[i], y[i])
        for rep in range(10):
            seed = 20263000 + rep * 100000 + int(i)
            t = time.perf_counter(); raw, _, _ = raw_scores(x[i], y[i], rank, nodes, seed); dt = time.perf_counter() - t
            errors["released"].append(raw[0] - ref); elapsed["released"] += dt
            errors["weighted_empirical"].append(raw[2] - ref); elapsed["weighted_empirical"] += dt
            t = time.perf_counter(); anchored, _, _ = anchored_score(x[i], y[i], rank, nodes, seed); elapsed["anchored_weighted"] += time.perf_counter() - t
            errors["anchored_weighted"].append(anchored - ref)
            rng = np.random.default_rng(seed)
            t = time.perf_counter(); subset = x[i, rng.choice(x.shape[1], nodes, replace=False)]; score = fair_score(subset, y[i]); elapsed["random_subsample_fair"] += time.perf_counter() - t
            errors["random_subsample_fair"].append(score - ref)
            t = time.perf_counter(); ordered = np.sort(x[i]); qidx = np.linspace(0, len(ordered) - 1, nodes).round().astype(int); score = empirical_score(ordered[qidx], y[i]); elapsed["quantile_empirical"] += time.perf_counter() - t
            errors["quantile_empirical"].append(score - ref)
    return {
        method: {"RMSE_vs_full_fair": float(np.sqrt(np.mean(np.square(errors[method])))), "seconds": elapsed[method], "microseconds_per_score": elapsed[method] / len(errors[method]) * 1e6}
        for method in methods
    }


def rolling_candidates(x, y, start=365):
    alpha_grid = np.array([0.6, 0.8, 1.0, 1.2, 1.4])
    all_candidates = []
    chosen_alpha = []
    for t in range(start, len(y)):
        hx, hy = x[t - start:t], y[t - start:t]
        bias = float(np.mean(hx.mean(axis=1) - hy))
        centered = hx - hx.mean(axis=1, keepdims=True)
        corrected_mean = hx.mean(axis=1, keepdims=True) - bias
        losses = []
        for alpha in alpha_grid:
            forecasts = corrected_mean + alpha * centered
            losses.append(np.mean([fair_score(forecasts[j], hy[j]) for j in range(start)]))
        alpha = float(alpha_grid[int(np.argmin(losses))])
        current = x[t]
        current_mean = current.mean()
        raw = current
        bias_only = current - bias
        affine = current_mean - bias + alpha * (current - current_mean)
        climatology = np.quantile(hy, np.linspace(0, 1, x.shape[1]))
        all_candidates.append(np.stack([raw, bias_only, affine, climatology]))
        chosen_alpha.append(alpha)
        if (t - start + 1) % 500 == 0:
            print(f"candidate construction: {t - start + 1}/{len(y) - start}", flush=True)
    return np.asarray(all_candidates), np.asarray(chosen_alpha)


def ranking_study(data):
    x, y = data["x"], data["y"]
    rank, nodes = data["config"]
    candidates, alphas = rolling_candidates(x, y)
    yy = y[365:]
    scores = np.full((len(yy), 2, 3, 4), np.nan)
    failures = []
    for i in range(len(yy)):
        for candidate in range(4):
            for oi, shift in enumerate(data["shifts"]):
                seed = 20264000 + i * 4 + candidate
                try:
                    raw, _, _ = raw_scores(candidates[i, candidate] + shift, yy[i] + shift, rank, nodes, seed)
                    anchored, _, _ = anchored_score(candidates[i, candidate] + shift, yy[i] + shift, rank, nodes, seed)
                    scores[i, oi, 0, candidate] = raw[0]
                    scores[i, oi, 1, candidate] = anchored
                    scores[i, oi, 2, candidate] = fair_score(candidates[i, candidate] + shift, yy[i] + shift)
                except Exception as exc:
                    failures.append({"row": i + 365, "candidate": candidate, "origin": oi, "error": repr(exc)})
        if (i + 1) % 500 == 0:
            print(f"ranking study: {i + 1}/{len(yy)}", flush=True)
    winners = np.nanargmin(scores, axis=3)
    changed = winners[:, 0] != winners[:, 1]
    methods = ["released", "anchored_weighted", "full_fair"]
    summary = {}
    for j, method in enumerate(methods):
        indicator = changed[:, j].astype(float)
        ci = {}
        for block in (7, 14, 30):
            rng = np.random.default_rng(20261010 + block)
            vals = np.empty(2000)
            for b in range(2000):
                vals[b] = indicator[moving_block_indices(len(indicator), block, rng)].mean()
            ci[f"moving_block_{block}"] = np.quantile(vals, [0.025, 0.975]).tolist()
        summary[method] = {"winner_changes": int(indicator.sum()), "proportion": float(indicator.mean()), "CI95": ci}
    return {"cases": len(yy), "candidate_order": ["raw", "bias_corrected", "affine_corrected", "rolling_climatology"], "alpha_counts": {str(a): int(np.sum(alphas == a)) for a in np.unique(alphas)}, "failures": failures, "summary": summary}


def main():
    data = load_data()
    output = {"source_commit": "dddd3ea46a6199d2ed48019c7825e79fb242787e", "methods": ["released", "normalized_pair", "weighted_empirical", "anchored_weighted", "full_fair"], "datasets": {}}
    for name, item in data.items():
        full, values = full_validation(name, item)
        np.save(ROOT / f"{name.lower().replace(' ', '_')}_expanded_scores.npy", values)
        output["datasets"][name] = {"full_validation": full}
        (ROOT / "expanded_validation_results.partial.json").write_text(json.dumps(output, indent=2))
    for name, item in data.items():
        output["datasets"][name]["seed_budget_sensitivity"] = sensitivity(name, item)
        output["datasets"][name]["equal_budget"] = equal_budget(name, item)
        (ROOT / "expanded_validation_results.partial.json").write_text(json.dumps(output, indent=2))
    output["ranking_study"] = ranking_study(data["Innsbruck"])
    output["environment"] = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    (ROOT / "expanded_validation_results.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
