"""Audit the full public commit history without redistributing source code."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "source_history_results.json"
REPOSITORY = "https://github.com/ma921/CRPS.git"
TARGET_PATHS = ("utils/crps.py", "utils/kernel_quadrature.py")


def run(*args, cwd=None):
    return subprocess.run(
        list(args), cwd=cwd, check=True, text=True, capture_output=True
    ).stdout.strip()


def audit_repository(repository: str):
    with tempfile.TemporaryDirectory(prefix="crps-history-audit-") as directory:
        run("git", "clone", "--quiet", repository, directory)
        commits = run(
            "git",
            "log",
            "--reverse",
            "--format=%H|%aI|%s",
            cwd=directory,
        ).splitlines()
        records = []
        for row in commits:
            commit, authored_at, subject = row.split("|", 2)
            paths = {}
            for path in TARGET_PATHS:
                present = path in run(
                    "git", "ls-tree", "-r", "--name-only", commit, cwd=directory
                ).splitlines()
                paths[path] = (
                    run("git", "rev-parse", f"{commit}:{path}", cwd=directory)
                    if present
                    else None
                )
            records.append(
                {
                    "commit": commit,
                    "authored_at": authored_at,
                    "subject": subject,
                    "target_blobs": paths,
                }
            )

        refs = []
        for row in run("git", "ls-remote", "--heads", repository).splitlines():
            commit, ref = row.split()
            refs.append({"ref": ref, "commit": commit})

    code_records = [
        item for item in records if item["target_blobs"]["utils/crps.py"] is not None
    ]
    crps_blobs = {item["target_blobs"]["utils/crps.py"] for item in code_records}
    kq_blobs = {
        item["target_blobs"]["utils/kernel_quadrature.py"] for item in code_records
    }
    return {
        "repository": repository,
        "audit_date": date.today().isoformat(),
        "commit_count": len(records),
        "branch_heads": refs,
        "commits": records,
        "code_present_from_commit": code_records[0]["commit"] if code_records else None,
        "commits_containing_code": len(code_records),
        "distinct_crps_blob_count": len(crps_blobs),
        "distinct_kernel_quadrature_blob_count": len(kq_blobs),
        "target_code_unchanged_after_introduction": bool(
            code_records and len(crps_blobs) == 1 and len(kq_blobs) == 1
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit_repository(args.repository)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
