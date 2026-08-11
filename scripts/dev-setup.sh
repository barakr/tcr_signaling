#!/usr/bin/env bash
#
# One-time setup for a fresh clone. Safe to re-run: every step checks first and
# reports "already correct" rather than changing anything.
#
#   ./scripts/dev-setup.sh
#
# Everything here lives in .git/, which is never committed — so none of it
# arrives with a clone, and all of it is silently absent until this runs. That
# is the point: the two settings below are not conveniences.
#
#   1. core.hooksPath   — without it NO hook runs. Not commit-msg, not
#                         pre-commit, not pre-push: no conventional-commit
#                         check, no ruff, no fast tests, no Status.md check.
#                         They do not fail; they simply never execute, and the
#                         repo looks green. This is the "a gate that can skip
#                         must be able to fail for skipping" failure (CLAUDE.md
#                         rule 12) sitting in the setup itself.
#
#   2. core.worktree    — when this repo is a git submodule, `git submodule`
#                         writes a *relative* core.worktree into the shared
#                         config. It is resolved against the gitdir, and a
#                         linked worktree's gitdir is two levels deeper, so the
#                         inherited value lands on the module directory, which
#                         holds no files. `git status` in any worktree then
#                         reports every tracked file as deleted, and
#                         `git commit -a` would record that deletion.
#                         core.worktree is a per-worktree setting; moving it out
#                         of the shared config is the documented git fix.
#                         models/kinetic_segregation/tests/test_checkout_health.py
#                         fails if this is ever undone.
#
set -euo pipefail

# Locate the repo from THIS SCRIPT's own path, never from
# `git rev-parse --show-toplevel`. In the broken state that command is exactly
# what is lying, so trusting it lands this script in the wrong directory and it
# then "verifies" somewhere the files are not. (An earlier version did precisely
# that — the same defect, in the same shape, as the shader lookup that started
# all this: resolve relative to yourself, not to ambient state.)
cd "$(dirname "$(cd "$(dirname "$0")" && pwd -P)")"
COMMON="$(git rev-parse --git-common-dir)"
[[ "$COMMON" = /* ]] || COMMON="$PWD/$COMMON"

changed=0
note() { printf '  %s\n' "$1"; }

echo "dev-setup: $PWD"   # $PWD, not --show-toplevel: see the cd above

# ── 1. Git hooks ────────────────────────────────────────────────────────────
echo
echo "git hooks (githooks/)"
# Test what the setting DOES, not how it is spelled. An absolute path is as
# valid as a relative one, and agent tooling writes an absolute one pointing at
# the main checkout — functionally fine, since it holds the same tracked hooks.
# A literal string comparison would call that broken and "fix" it, and a setup
# script that misreports what it changed is the same class of problem as the
# ones it exists to prevent.
hooks_cfg="$(git config --get core.hooksPath || true)"
if [[ -n "$hooks_cfg" && "$hooks_cfg" != /* ]]; then
    hooks_abs="$PWD/$hooks_cfg"
else
    hooks_abs="$hooks_cfg"
fi
if [[ -n "$hooks_abs" && -x "$hooks_abs/pre-commit" ]]; then
    note "already correct (hooks run from $hooks_cfg)"
else
    # Write where it will actually take effect. With extensions.worktreeConfig
    # enabled, a per-worktree value shadows the shared one, so setting the
    # shared one would change nothing while reporting success.
    if git config --worktree --get core.hooksPath >/dev/null 2>&1; then
        git config --worktree core.hooksPath githooks
    else
        git config core.hooksPath githooks
    fi
    note "SET core.hooksPath=githooks — no hook was running before this"
    changed=1
fi

# ── 2. Worktree resolution ──────────────────────────────────────────────────
echo
echo "worktree resolution"
shared_wt="$(git config --file "$COMMON/config" --get core.worktree || true)"
if [[ -z "$shared_wt" ]]; then
    note "already correct (nothing inherited from the shared config)"
else
    # The extension is what makes a per-worktree config file be read at all.
    if [[ "$(git config --get extensions.worktreeConfig || true)" != "true" ]]; then
        git config extensions.worktreeConfig true
        note "enabled extensions.worktreeConfig"
    fi
    # Keep the value verbatim: the relative form is correct for the main
    # worktree and keeps the tree relocatable. Only its *inheritance* was wrong.
    git config --file "$COMMON/config.worktree" core.worktree "$shared_wt"
    git config --file "$COMMON/config" --unset core.worktree
    note "MOVED core.worktree out of the shared config (was: $shared_wt)"
    note "new worktrees will now resolve themselves correctly"
    changed=1
fi

# ── Verify, rather than assume ──────────────────────────────────────────────
echo
echo "verifying"
top="$(git rev-parse --show-toplevel)"
if [[ "$top" != "$PWD" ]]; then
    echo "  FAILED: git still reports this checkout at $top, not $PWD" >&2
    exit 1
fi
phantom="$(git status --porcelain | awk '$1 ~ /D/ {print $2}' | while read -r f; do
    [[ -e "$f" ]] && echo "$f"
done | wc -l | tr -d ' ')"
if [[ "$phantom" != "0" ]]; then
    echo "  FAILED: git calls $phantom present file(s) deleted" >&2
    exit 1
fi
note "git agrees with the filesystem"

echo
if [[ "$changed" == "1" ]]; then
    echo "dev-setup: done — settings were missing and have been applied."
else
    echo "dev-setup: done — nothing needed changing."
fi
