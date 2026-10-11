import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd):
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def ensure_pinned_source():
    run([sys.executable, "real_data/fetch_pinned_source.py"])


def check_expanded_results():
    path = ROOT / "real_data" / "expanded_validation_results.json"
    data = json.loads(path.read_text())
    inns = data["datasets"]["Innsbruck"]["full_validation"]
    pnw = data["datasets"]["Pacific Northwest"]["full_validation"]
    ranking = data["ranking_study"]
    assert inns["rows"] == 2749
    assert pnw["rows"] == 36826
    assert inns["controls"]["pass"] is True
    assert pnw["controls"]["pass"] is True
    assert inns["summary"]["released"]["proportion_above_0_01"] == 1.0
    assert pnw["summary"]["released"]["proportion_above_0_01"] == 1.0
    assert inns["summary"]["anchored_weighted"]["maximum"] <= 2e-4
    assert pnw["summary"]["anchored_weighted"]["maximum"] <= 2e-4
    assert ranking["summary"]["released"]["winner_changes"] == 1624
    assert ranking["summary"]["anchored_weighted"]["winner_changes"] == 0
    assert ranking["summary"]["full_fair"]["winner_changes"] == 0
    print("expanded validation checks passed")


def check_affine_extension():
    path = ROOT / "real_data" / "affine_extension_results.json"
    data = json.loads(path.read_text())
    precipitation = data["precipitation"]
    controls = precipitation["controls"]
    runtime = data["runtime_accuracy"]
    assert precipitation["rows"] == 2749
    assert precipitation["ensemble_members"] == 11
    assert controls["failures"] == []
    assert controls["constant_ensemble_rows"] == 64
    assert controls["pass"] is True
    assert controls["affine_standardized_maximum_mm"] <= 2e-4
    assert controls["full_fair_maximum_mm"] <= 2e-4
    rows = runtime["rows"]
    assert sorted({row["ensemble_size"] for row in rows}) == [100, 500, 2000]
    assert len(rows) == 12
    assert all(row["repetitions"] == 12 for row in rows)
    methods = {row["method"] for row in rows}
    assert methods == {
        "released_compressed",
        "affine_standardized",
        "full_fair_sorted",
        "full_fair_pairwise",
    }
    print("affine extension checks passed")


def check_independent_controls():
    path = ROOT / "independent_controls" / "cross_library_results.json"
    data = json.loads(path.read_text())
    assert data["total_forecast_cases"] == 4000
    assert data["transformations_per_case"] == 5
    assert data["absolute_tolerance_in_original_units"] == 1e-8
    all_rows = list(data["equal_weight"].values()) + list(
        data["unequal_weight"].values()
    )
    assert all(row["comparisons"] == 20000 for row in all_rows)
    assert all(row["failures"] == 0 for row in all_rows)
    assert max(row["maximum_mapped_back_error"] for row in all_rows) < 2e-12

    history_path = ROOT / "independent_controls" / "source_history_results.json"
    history = json.loads(history_path.read_text())
    assert history["commit_count"] == 4
    assert history["commits_containing_code"] == 3
    assert history["distinct_crps_blob_count"] == 1
    assert history["distinct_kernel_quadrature_blob_count"] == 1
    assert history["target_code_unchanged_after_introduction"] is True
    print("independent control and source history checks passed")


def check_bike_sharing_extension():
    path = ROOT / "real_data" / "bike_sharing_affine_results.json"
    data = json.loads(path.read_text())
    assert data["source_rows"] == 17379
    assert data["eligible_past_only_cases"] == 16419
    assert data["audited_cases"] == 2000
    assert data["forecast"]["uses_future_data"] is False
    assert data["controls"]["failures"] == []
    assert data["controls"]["pass"] is True
    assert data["controls"]["affine_standardized_maximum_rides"] < 3e-13
    diagnostic = data["development_diagnostic"]
    assert diagnostic["legacy_half_grid_case_transform_mismatches"] == 10
    assert diagnostic["legacy_maximum_score_residual_rides"] > 0.4
    assert diagnostic["guarded_half_up_case_transform_mismatches"] == 0
    scale = data["summary"]["rides_to_hundreds"]["released_compressed"]
    stress = data["summary"]["affine_stress_recoding"]["released_compressed"]
    assert scale["proportion_above_tolerance"] == 0.4795
    assert stress["proportion_above_tolerance"] == 1.0
    print("bike-sharing extension checks passed")


def write_checksums():
    targets = [
        ROOT / "manuscript_simple_revised.tex",
        ROOT / "manuscript_simple_revised.pdf",
        ROOT / "real_data" / "expanded_validation_results.json",
        ROOT / "real_data" / "real_data_results.json",
        ROOT / "real_data" / "affine_extension_results.json",
        ROOT / "real_data" / "precipitation_affine_summary.csv",
        ROOT / "real_data" / "runtime_accuracy_summary.csv",
        ROOT / "real_data" / "run_expanded_validation.py",
        ROOT / "real_data" / "run_real_data_audit.py",
        ROOT / "real_data" / "run_affine_extension.py",
        ROOT / "real_data" / "run_bike_sharing_extension.py",
        ROOT / "real_data" / "corrected_crps.py",
        ROOT / "real_data" / "bike_sharing_affine_results.json",
        ROOT / "real_data" / "fetch_pinned_source.py",
        ROOT / "tests" / "test_affine_wrapper.py",
        ROOT / "requirements-main-lock.txt",
        ROOT / "requirements-real-data-lock.txt",
        ROOT / "requirements-independent-controls-lock.txt",
        ROOT / "DATA_SOURCES.md",
        ROOT / "CITATION.cff",
        ROOT / ".zenodo.json",
        ROOT / "supporting" / "area013_distribution_budget_robustness.py",
        ROOT / "supporting" / "area013_diagnose_float32_translation.py",
        ROOT / "supporting" / "area013_f13_10_results.json",
        ROOT / "supporting" / "area013_f13_10_diagnosis_results.json",
        ROOT / "independent_controls" / "run_cross_library_controls.py",
        ROOT / "independent_controls" / "cross_library_results.json",
        ROOT / "independent_controls" / "audit_source_history.py",
        ROOT / "independent_controls" / "source_history_results.json",
        ROOT / "independent_controls" / "README.md",
    ]
    lines = []
    for target in targets:
        if target.exists():
            lines.append(f"{sha256(target)}  {target.relative_to(ROOT).as_posix()}")
    out = ROOT / "SHA256SUMS.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def main():
    ensure_pinned_source()
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "manuscript_simple_revised.tex"])
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "manuscript_simple_revised.tex"])
    check_expanded_results()
    check_affine_extension()
    check_bike_sharing_extension()
    check_independent_controls()
    write_checksums()


if __name__ == "__main__":
    main()
