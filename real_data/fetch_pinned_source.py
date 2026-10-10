"""Fetch and verify the exact public source audited by this study."""

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "author_crps"
REPOSITORY = "https://github.com/ma921/CRPS.git"
COMMIT = "dddd3ea46a6199d2ed48019c7825e79fb242787e"


def run(*args, cwd=None):
    return subprocess.run(
        list(args), cwd=cwd, check=True, text=True, capture_output=True
    ).stdout.strip()


def main():
    if TARGET.exists():
        if not (TARGET / ".git").exists():
            raise RuntimeError(
                f"{TARGET} exists but is not a Git checkout; move it aside first"
            )
        current = run("git", "rev-parse", "HEAD", cwd=TARGET)
        if current == COMMIT:
            print(f"verified existing source checkout at {COMMIT}")
            return
        raise RuntimeError(
            f"{TARGET} is at {current}, not the required commit {COMMIT}"
        )

    if shutil.which("git") is None:
        raise RuntimeError("Git is required to fetch the pinned public source")

    run("git", "clone", "--filter=blob:none", REPOSITORY, str(TARGET))
    run("git", "checkout", "--detach", COMMIT, cwd=TARGET)
    current = run("git", "rev-parse", "HEAD", cwd=TARGET)
    if current != COMMIT:
        raise RuntimeError(f"checked out {current}, expected {COMMIT}")
    print(f"fetched and verified public source commit {current}")


if __name__ == "__main__":
    main()
