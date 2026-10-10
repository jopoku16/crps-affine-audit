import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "author_crps"))
from utils.kernel_quadrature import (  # noqa: E402
    error_terms_kq_comp,
    indicator_kernel,
    kq_compress,
)

DATA = pd.read_csv(ROOT / "temp.csv")
M, RANK, NODES = 11, 8, 5
SHIFT = 273.15


def full_fair(x, y):
    x = np.asarray(x, dtype=np.float64)
    return np.mean(np.abs(x - y)) - np.abs(x[:, None] - x[None, :]).sum() / (2 * M * (M - 1))


def compressed_values(w, z, y):
    w = w.double()
    z = z.double()
    n = w / w.sum()
    h = indicator_kernel(z, z)
    distances = (z[:, None] - z[None, :]).abs()
    released = w @ (z - y).abs() + w @ z - 2 * NODES / (NODES - 1) * (w @ h @ w)
    q = n @ n
    normalized_pair = n @ (z - y).abs() + n @ z - 2 / (1 - q) * (n @ h @ n)
    weighted_empirical = n @ (z - y).abs() - 0.5 * (n @ distances @ n)
    return np.array([released.item(), normalized_pair.item(), weighted_empirical.item()])


scores = np.empty((len(DATA), 2, 4), dtype=np.float64)
mass_error = 0.0
minimum_weight = np.inf
public_error = 0.0
failures = []

for row_index, row in DATA.iterrows():
    y0 = float(row.iloc[0])
    x0 = row.iloc[1:].to_numpy(dtype=np.float64)
    scores[row_index, :, 3] = [full_fair(x0, y0), full_fair(x0 + SHIFT, y0 + SHIFT)]
    for origin_index, shift in enumerate((0.0, SHIFT)):
        seed = 20261008 + row_index
        x = torch.tensor(x0 + shift, dtype=torch.float32)
        y = float(np.float32(y0 + shift))
        try:
            torch.manual_seed(seed)
            w, z = kq_compress(x, RANK, NODES)
            vals = compressed_values(w, z, y)
            scores[row_index, origin_index, :3] = vals
            mass_error = max(mass_error, abs(float(w.double().sum()) - 1.0))
            minimum_weight = min(minimum_weight, float(w.min()))
            if row_index < 100:
                torch.manual_seed(seed)
                public = float(error_terms_kq_comp(x, y, n_low_rank=RANK, n_compressed=NODES))
                public_error = max(public_error, abs(public - vals[0]))
        except Exception as exc:
            failures.append({"row": int(row_index), "origin": origin_index, "error": repr(exc)})
            scores[row_index, origin_index, :3] = np.nan

if failures:
    raise RuntimeError(f"Source failures: {failures[:5]}")

names = ["released", "normalized_pair", "weighted_empirical", "full_fair"]
changes = np.abs(scores[:, 1] - scores[:, 0])
primary = {
    name: {
        "median": float(np.median(changes[:, i])),
        "p95": float(np.quantile(changes[:, i], 0.95)),
        "maximum": float(np.max(changes[:, i])),
        "count_above_0_01": int(np.sum(changes[:, i] > 0.01)),
        "practical_invariance_pass": bool(np.max(changes[:, i]) <= 2e-4),
    }
    for i, name in enumerate(names)
}

reference = scores[:, :, 3]
errors = scores - reference[:, :, None]
rmse = np.sqrt(np.mean(errors**2, axis=0))
rng = np.random.default_rng(20261009)
boot = rng.integers(0, len(DATA), size=(2000, len(DATA)))
secondary = {}
for method_index, name in enumerate(names[:3]):
    diff = np.sqrt(np.mean(errors[boot, :, method_index] ** 2, axis=1)) - np.sqrt(
        np.mean(errors[boot, :, 0] ** 2, axis=1)
    )
    secondary[name] = []
    for origin_index, origin in enumerate(("Celsius", "Kelvin")):
        secondary[name].append(
            {
                "origin": origin,
                "rmse": float(rmse[origin_index, method_index]),
                "difference_vs_released": float(rmse[origin_index, method_index] - rmse[origin_index, 0]),
                "difference_CI95": np.quantile(diff[:, origin_index], [0.025, 0.975]).tolist(),
            }
        )

controls = {
    "rows": len(DATA),
    "missing_values": int(DATA.isna().sum().sum()),
    "source_failures": len(failures),
    "maximum_weight_mass_error": mass_error,
    "minimum_weight": minimum_weight,
    "maximum_public_function_error": public_error,
    "maximum_full_fair_shift_error": float(np.max(changes[:, 3])),
}
controls["pass"] = bool(
    controls["missing_values"] == 0
    and controls["source_failures"] == 0
    and mass_error <= 2e-4
    and minimum_weight >= 0
    and public_error <= 2e-4
    and controls["maximum_full_fair_shift_error"] <= 1e-10
)

result = {
    "dataset": "ensemblepp 1.0-0 temp",
    "source_commit": "dddd3ea46a6199d2ed48019c7825e79fb242787e",
    "configuration": {"members": M, "landmarks": RANK, "compressed_nodes": NODES, "shift": SHIFT},
    "controls": controls,
    "primary": primary,
    "secondary": secondary,
}
(ROOT / "real_data_results.json").write_text(json.dumps(result, indent=2))
pd.DataFrame({f"{name}_{origin}": scores[:, oi, mi] for mi, name in enumerate(names) for oi, origin in enumerate(("C", "K"))}).to_csv(ROOT / "real_data_scores.csv", index=False)
print(json.dumps(result, indent=2))
