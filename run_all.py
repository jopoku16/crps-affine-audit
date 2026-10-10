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
        ROOT / "real_data" / "corrected_crps.py",
        ROOT / "real_data" / "fetch_pinned_source.py",
        ROOT / "tests" / "test_affine_wrapper.py",
        ROOT / "requirements-main-lock.txt",
        ROOT / "requirements-real-data-lock.txt",
        ROOT / "DATA_SOURCES.md",
        ROOT / "CITATION.cff",
        ROOT / ".zenodo.json",
        ROOT / "supporting" / "area013_distribution_budget_robustness.py",
        ROOT / "supporting" / "area013_diagnose_float32_translation.py",
        ROOT / "supporting" / "area013_f13_10_results.json",
        ROOT / "supporting" / "area013_f13_10_diagnosis_results.json",
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
    write_checksums()


if __name__ == "__main__":
    main()
