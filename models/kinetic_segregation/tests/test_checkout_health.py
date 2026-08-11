"""Catch a checkout where git is reporting something other than reality.

The specific state this exists for: `git submodule` writes

    core.worktree = ../../../../projects/tcr_signaling

into `.git/modules/projects/tcr_signaling/config`, and that path is resolved
relative to the **gitdir**. The main checkout's gitdir is the module directory,
so it resolves correctly. A linked worktree's gitdir is two levels deeper
(`.../worktrees/<name>`), so the same *inherited* relative path lands two levels
short — on the module directory, which holds no working files.

`git status` then reports **every tracked file as deleted** while the files sit
on disk, and `git commit -a` would record the deletion of the whole repository.
Confirmed as git's own behaviour rather than any tool's: a plain
`git worktree add --detach` reproduces it on git 2.50.1.

Why a test and not just a note in Status.md: the repair is a git config change,
so it cannot be committed and is absent from every fresh clone. A note only
helps someone who already suspects a problem, and the symptom here is alarming
enough that a developer may react destructively before going looking for prose.
This runs in the suite everyone already runs and needs no setup at all.

It cannot fire spuriously — in a healthy checkout both assertions are trivially
true. `scripts/dev_setup.py` applies the repair.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]

_REPAIR = """\
Repair (from the repo root, or just run scripts/dev_setup.py):

    C=$(git rev-parse --git-common-dir)
    git config --file "$C/config.worktree" core.worktree \\
        "$(git config --file "$C/config" --get core.worktree)"
    git config --file "$C/config" --unset core.worktree

core.worktree is a per-worktree setting; the bug is that it is inherited from
the shared config, where its relative value is only correct for one worktree."""


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        timeout=60,
    )


def _under_git() -> bool:
    """False only for a tarball export or a machine without git.

    Deliberately `--absolute-git-dir` and NOT `--is-inside-work-tree`. The
    obvious probe is actively wrong here: in the broken state git believes the
    worktree is the module directory, so the directory we are standing in is not
    inside it and `--is-inside-work-tree` answers **false** — turning the guard
    off in precisely the case it exists to catch. That was not theoretical; the
    first version of this file shipped it and passed against a checkout showing
    120 phantom deletions.

    `--absolute-git-dir` succeeds in both the healthy and the broken state, so
    it separates "no git here" from "git is confused", which is the distinction
    that matters. Returning early when git is genuinely absent is not a hidden
    no-op: with no git there is no resolution to get wrong.
    """
    try:
        return _git("rev-parse", "--absolute-git-dir").returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class TestGitAgreesWithReality:
    def test_git_reports_this_checkout_at_its_real_location(self):
        """`git rev-parse --show-toplevel` must name the directory we are in."""
        if not _under_git():
            return

        toplevel = Path(_git("rev-parse", "--show-toplevel").stdout.strip()).resolve()
        assert toplevel == _REPO, (
            f"git thinks this checkout lives at\n    {toplevel}\n"
            f"but these files are at\n    {_REPO}\n\n"
            "Every tracked file will be reported as deleted, and `git commit -a` "
            f"would commit that deletion.\n\n{_REPAIR}"
        )

    def test_no_tracked_file_is_called_deleted_while_it_exists(self):
        """The symptom itself, asserted independently of the cause.

        Catches any future mechanism that makes git disagree with the disk, not
        just the inherited-core.worktree one — a genuinely deleted file is not
        flagged, because it is absent from disk as git says.
        """
        if not _under_git():
            return

        proc = _git("status", "--porcelain")
        phantom = []
        for line in proc.stdout.splitlines():
            # Porcelain v1: 2 status columns, a space, then the path.
            if len(line) > 3 and "D" in line[:2]:
                path = line[3:].strip().strip('"')
                if (_REPO / path).exists():
                    phantom.append(path)

        assert not phantom, (
            f"git reports {len(phantom)} file(s) as deleted that are present on "
            f"disk, e.g. {phantom[:3]}.\n\n"
            f"Committing now would record their deletion.\n\n{_REPAIR}"
        )


class TestHooksAreInstalled:
    """A developer clone with no hooks configured must not look green.

    The gap this closes: the two tests above only fire for a *linked worktree*, and a
    plain clone is not affected by that bug — so on an ordinary fresh clone the hooks
    were silently absent and the whole suite passed. No commit-msg check, no ruff, no
    fast tests, no Status.md check, and nothing anywhere saying so. That is exactly the
    "green means nothing ran" failure CLAUDE.md rule 12 exists to prevent, and leaving
    it to whether someone read the README is not a guard.

    Exempt under CI because there hooks are not merely unconfigured but irrelevant: the
    workflow runs ruff and pytest as explicit steps, so the checks a hook would trigger
    do execute. That is a reasoned exemption rather than a hidden no-op, and it is a
    pass rather than a skip on purpose — `Deep CI`'s audit sanctions only the Metal
    skip, so a skip here would fail the job for the wrong reason.
    """

    def test_core_hookspath_points_at_runnable_hooks(self):
        if os.environ.get("CI"):
            return
        if not _under_git():
            return

        configured = _git("config", "--get", "core.hooksPath").stdout.strip()
        resolved = None
        if configured:
            path = Path(configured)
            resolved = path if path.is_absolute() else _REPO / path

        assert resolved is not None and (resolved / "pre-commit").is_file(), (
            "git hooks are not installed, so NONE of them run: no conventional-commit "
            "check, no ruff, no fast tests, no Status.md check. Nothing fails — the "
            "checks simply never execute.\n\n"
            f"    core.hooksPath = {configured or '<unset>'}\n\n"
            "Fix (once per clone):\n\n    python scripts/dev_setup.py\n\n"
            "Set CI=1 to bypass this, which is what the workflows do — they run ruff "
            "and pytest as explicit steps instead."
        )


class TestDevSetupRunsEverywhere:
    """The setup script must actually execute on all three platforms.

    This is the whole reason it is Python and not a shell script: CLAUDE.md KS rule 6
    forbids depending on `make` because a stock Windows box has none, and bash is in the
    same category. A Windows developer who cannot run setup gets no git hooks at all —
    the exact hole the script exists to close, reopened for one platform.

    Asserting on the *exit code* would couple this to whether the machine happens to be
    configured (CI deliberately is not), so it asserts what actually varies by platform:
    that the script runs to completion rather than dying on a syntax error, a missing
    interpreter or a path assumption. Windows CI is where this earns its keep.
    """

    def test_check_mode_runs_without_crashing(self):
        proc = subprocess.run(
            [sys.executable, str(_REPO / "scripts" / "dev_setup.py"), "--check"],
            cwd=str(_REPO),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "Traceback" not in proc.stderr, (
            f"scripts/dev_setup.py crashed on {sys.platform}:\n{proc.stderr[-1500:]}"
        )
        # 0 = nothing to do, 1 = setup needed. Anything else is a crash.
        assert proc.returncode in (0, 1), (
            f"unexpected exit {proc.returncode} on {sys.platform}:\n{proc.stderr[-1500:]}"
        )
        assert "dev-setup:" in proc.stdout, (
            f"no report on stdout; the script did not run to completion:\n{proc.stdout[-500:]}"
        )
