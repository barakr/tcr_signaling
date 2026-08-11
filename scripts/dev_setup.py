#!/usr/bin/env python3
"""One-time setup for a fresh clone. Safe to re-run.

    python scripts/dev_setup.py            # apply what is missing
    python scripts/dev_setup.py --check    # report only; exit 1 if work is needed

Python rather than a shell script for the same reason CLAUDE.md KS rule 6 forbids
depending on `make`: a stock Windows box has neither. Bash is in that category too, and
the consequence here is worse than an awkward build — a Windows developer who cannot run
setup gets no git hooks at all, which is precisely the hole this script exists to close.
Python >= 3.12 is already required by everything else in the repo, and `ks_build.py` is
the precedent for a portable helper carrying no framework dependency.

Everything configured here lives in `.git/`, which is never committed, so none of it
arrives with a clone and all of it is silently absent until this runs:

* **core.hooksPath** — without it NO hook runs. Not commit-msg, not pre-commit, not
  pre-push: no conventional-commit check, no ruff, no fast tests, no Status.md check.
  Nothing fails; the checks simply never execute and the repo looks green. That is
  CLAUDE.md rule 12's own failure mode sitting in the setup.

* **core.worktree** — when this repo is a git submodule, `git submodule` writes a
  *relative* core.worktree into the shared config. It resolves against the gitdir, and a
  linked worktree's gitdir is two levels deeper, so the inherited value lands on the
  module directory, which holds no files. `git status` in any worktree then reports every
  tracked file as deleted, and `git commit -a` would record that deletion. core.worktree
  is a per-worktree setting; moving it out of the shared config is the documented fix.

`models/kinetic_segregation/tests/test_checkout_health.py` fails if the second is undone.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Resolve the repo from THIS FILE's own location, never from
# `git rev-parse --show-toplevel`. In the broken state that command is exactly what is
# lying, so trusting it lands this script in the wrong directory and it then "verifies"
# somewhere the files are not. An earlier shell version did precisely that — the same
# defect, in the same shape, as the Metal shader lookup that prompted this work.
REPO = Path(__file__).resolve().parents[1]


def git(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=60,
        check=check,
    )


def config_get(*args: str) -> str | None:
    """A git config value, or None when unset (exit 1, which is not an error here)."""
    proc = git("config", *args)
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else None


def common_dir() -> Path:
    raw = git("rev-parse", "--git-common-dir").stdout.strip()
    path = Path(raw)
    return path if path.is_absolute() else (REPO / path).resolve()


def check_hooks(apply: bool) -> tuple[bool, str]:
    """Ensure git hooks actually run. Returns (needs_work, message).

    Tests what the setting DOES, not how it is spelled: an absolute path is as valid as a
    relative one, and agent tooling writes an absolute one pointing at the main checkout,
    which holds the same tracked hooks. A string comparison would call that broken and
    "fix" it — and a setup script that misreports what it changed is the same class of
    problem as the ones it exists to prevent.
    """
    configured = config_get("--get", "core.hooksPath")
    if configured:
        path = Path(configured)
        resolved = path if path.is_absolute() else REPO / path
        if (resolved / "pre-commit").is_file():
            return False, f"already correct (hooks run from {configured})"

    if not apply:
        return True, "NOT SET — no git hook is running"

    # Write at the scope that takes effect. With extensions.worktreeConfig enabled a
    # per-worktree value shadows the shared one, so setting the shared one would change
    # nothing while reporting success.
    scope = ["--worktree"] if config_get("--worktree", "--get", "core.hooksPath") else []
    git("config", *scope, "core.hooksPath", "githooks", check=True)
    return True, "SET core.hooksPath=githooks — no hook was running before this"


def check_worktree(apply: bool) -> tuple[bool, str]:
    """Move an inherited core.worktree out of the shared config."""
    common = common_dir()
    shared = config_get("--file", str(common / "config"), "--get", "core.worktree")
    if not shared:
        return False, "already correct (nothing inherited from the shared config)"

    if not apply:
        return True, f"INHERITED core.worktree={shared} — every worktree will mis-resolve"

    if config_get("--get", "extensions.worktreeConfig") != "true":
        git("config", "extensions.worktreeConfig", "true", check=True)

    # Keep the value verbatim: the relative form is correct for the main worktree and
    # keeps the tree relocatable. Only its *inheritance* was ever wrong.
    git("config", "--file", str(common / "config.worktree"), "core.worktree", shared, check=True)
    git("config", "--file", str(common / "config"), "--unset", "core.worktree", check=True)
    return True, f"MOVED core.worktree out of the shared config (was: {shared})"


def verify() -> list[str]:
    """Confirm git now agrees with the filesystem. Returns a list of problems."""
    problems = []
    toplevel = git("rev-parse", "--show-toplevel").stdout.strip()
    if toplevel and Path(toplevel).resolve() != REPO:
        problems.append(f"git reports this checkout at {toplevel}, not {REPO}")

    phantom = [
        line[3:].strip().strip('"')
        for line in git("status", "--porcelain").stdout.splitlines()
        if len(line) > 3 and "D" in line[:2]
    ]
    present = [p for p in phantom if (REPO / p).exists()]
    if present:
        problems.append(f"git calls {len(present)} present file(s) deleted, e.g. {present[:3]}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what is missing and exit 1 if anything is, without changing it",
    )
    args = parser.parse_args()
    apply = not args.check

    print(f"dev-setup: {REPO}")
    if git("rev-parse", "--absolute-git-dir").returncode != 0:
        print("  not a git checkout — nothing to configure")
        return 0

    needs_work = False
    for title, fn in (
        ("git hooks (githooks/)", check_hooks),
        ("worktree resolution", check_worktree),
    ):
        print(f"\n{title}")
        work, message = fn(apply)
        needs_work = needs_work or work
        print(f"  {message}")

    print("\nverifying")
    problems = verify()
    for problem in problems:
        print(f"  FAILED: {problem}", file=sys.stderr)
    if problems:
        if args.check:
            print("\ndev-setup: run without --check to repair.")
        return 1
    print("  git agrees with the filesystem")

    print()
    if args.check:
        print("dev-setup: work needed." if needs_work else "dev-setup: nothing to do.")
        return 1 if needs_work else 0
    print(
        "dev-setup: done — settings were missing and have been applied."
        if needs_work
        else "dev-setup: done — nothing needed changing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
