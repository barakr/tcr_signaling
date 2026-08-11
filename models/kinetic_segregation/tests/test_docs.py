"""Keep the install/build documentation from drifting out of truth.

Build instructions live in several places by necessity — a reader landing on the
repo root, a reader browsing the model directory, and a reader whose build just
failed all need them, and none of them will follow a link chain. Duplication is
therefore fine; *unverified* duplication is not.

This is the file that noticed the last drift too late: `models/kinetic_segregation/
README.md` said "Requires macOS with Command Line Tools" for months after the model
built on Linux, and it was still saying it after Windows support landed. Nothing
could have caught that, because nothing compared the docs to anything.

So there is exactly one canonical prose section, and these tests pin everything
else to a checkable source:

* build commands  -> compared against what `KS model CI` actually runs
* install commands -> compared against what `ks_build` prints on a failed build
* platform claims  -> no doc may say the build needs macOS

The rule for adding docs: if you write a build or install command into markdown,
it must either match one of those sources or link to the canonical section.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from models.kinetic_segregation import ks_build

_PKG = Path(__file__).resolve().parents[1]
_REPO = _PKG.parents[1]
_WORKFLOW = _REPO / ".github" / "workflows" / "ci.yml"

# The single canonical place for per-platform install instructions.
_CANONICAL_DOC = _REPO / "README.md"
_CANONICAL_HEADING = "## Build prerequisites (Windows, macOS, Linux)"
# GitHub's slug for that heading; what every other doc links to.
_CANONICAL_ANCHOR = "build-prerequisites-windows-macos-linux"


def compiled_models() -> list[Path]:
    """Model directories that compile something, found rather than listed.

    A model is "compiled" if it has a CMakeLists.txt. Kinetic segregation is the
    only one today; the point of discovering them is that the second one is held
    to the same documentation standard on the day it lands, without anyone
    remembering to extend this file.

    (When a second compiled model does appear, move this module to
    `models/tests/` — it is only here because there is currently one.)
    """
    models_dir = _REPO / "models"
    return sorted(
        p for p in models_dir.iterdir() if p.is_dir() and (p / "CMakeLists.txt").is_file()
    )


def _model_readmes() -> list[Path]:
    return [p / "README.md" for p in compiled_models() if (p / "README.md").is_file()]


def _docs() -> list[Path]:
    """Markdown a user might read, excluding the append-only decision log.

    Status.md is deliberately out of scope: it is a historical record, and old
    entries legitimately describe how things used to be.
    """
    candidates = [
        _REPO / "README.md",
        _REPO / "CLAUDE.md",
        _REPO / "notebooks" / "README.md",
        *_model_readmes(),
    ]
    return [p for p in candidates if p.is_file()]


def _canonical_section() -> str:
    text = _CANONICAL_DOC.read_text(encoding="utf-8")
    start = text.find(_CANONICAL_HEADING)
    assert start != -1, (
        f"{_CANONICAL_DOC.name} must contain the canonical heading "
        f"{_CANONICAL_HEADING!r}; every other doc links to it"
    )
    nxt = text.find("\n## ", start + len(_CANONICAL_HEADING))
    return text[start : nxt if nxt != -1 else len(text)]


def _cmake_lines(text: str) -> list[str]:
    """Every `cmake ...` invocation in a blob, whitespace-normalised."""
    out = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("$").strip()
        if line.startswith("cmake "):
            out.append(" ".join(line.split()))
    return out


class TestCanonicalSection:
    def test_it_exists_and_names_all_three_platforms(self):
        section = _canonical_section()
        for platform in ("Windows", "macOS", "Linux"):
            assert platform in section, (
                f"the canonical build section no longer mentions {platform}; "
                "it is the one place a user is sent to for install commands"
            )

    def test_other_docs_link_to_it_rather_than_re_explaining(self):
        """Entry points must point at the canonical section, not re-describe it.

        Every compiled model's README is included automatically, so a model added
        later inherits the rule instead of quietly inventing its own install prose.
        """
        linkers = [_REPO / "notebooks" / "README.md", *_model_readmes()]
        for doc in linkers:
            assert _CANONICAL_ANCHOR in doc.read_text(encoding="utf-8"), (
                f"{doc.relative_to(_REPO)} should link to "
                f"#{_CANONICAL_ANCHOR} instead of describing installation itself"
            )

    def test_the_anchor_matches_the_heading(self):
        """A link to a heading that was renamed is a silent 404 on GitHub."""
        slug = _CANONICAL_HEADING.removeprefix("## ").lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"\s+", "-", slug.strip())
        assert slug == _CANONICAL_ANCHOR, (
            f"heading {_CANONICAL_HEADING!r} slugs to {slug!r}, but docs link to "
            f"{_CANONICAL_ANCHOR!r}. Rename the links or restore the heading."
        )


class TestDocsMatchCI:
    def test_ci_builds_with_the_documented_commands(self):
        """The commands a reader is told to run must be the ones CI proves work.

        Parsed from the workflow rather than hardcoded here, so this fails when
        either side moves — which is the whole point.
        """
        yaml = pytest.importorskip("yaml")
        spec = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
        steps = spec["jobs"]["ks-model"]["steps"]
        build_steps = [s for s in steps if s.get("name", "").startswith("Build KS model")]
        assert build_steps, "no 'Build KS model' step in the workflow"

        ci_cmds = _cmake_lines(build_steps[0]["run"])
        assert ci_cmds == list(ks_build.BUILD_COMMANDS), (
            f"CI builds with {ci_cmds}, but ks_build.BUILD_COMMANDS says "
            f"{list(ks_build.BUILD_COMMANDS)}. One of them is wrong."
        )

    def test_the_canonical_section_shows_the_real_build_commands(self):
        documented = _cmake_lines(_canonical_section())
        for cmd in ks_build.BUILD_COMMANDS:
            assert cmd in documented, (
                f"README's build section does not show `{cmd}`. It must show the "
                "same commands CI runs, or a reader is following instructions "
                "nothing tests."
            )

    @pytest.mark.parametrize("readme", _model_readmes(), ids=lambda p: p.parent.name)
    def test_every_compiled_model_documents_how_to_build_it(self, readme: Path):
        """Generic on purpose: any future compiled model must show its own commands.

        Not pinned to kinetic segregation's exact two lines, because another model
        may configure differently — only that a reader browsing that directory is
        given a real `cmake` invocation rather than nothing.
        """
        documented = _cmake_lines(readme.read_text(encoding="utf-8"))
        assert any(c.startswith("cmake -S") for c in documented), (
            f"{readme.relative_to(_REPO)} shows no `cmake -S ...` command; someone "
            "browsing that directory on GitHub has no way to build it"
        )


class TestDocsMatchCode:
    def test_windows_install_command_matches_what_a_failed_build_prints(self):
        """`toolchain_hint()` and the README must not describe different installs.

        Checked from any platform: the hints are a dict precisely so a Mac can
        verify the Windows text.
        """
        winget_in_code = [
            " ".join(ln.split())
            for ln in ks_build.TOOLCHAIN_HINTS["windows"].splitlines()
            if "winget install Microsoft.VisualStudio" in ln
        ]
        assert winget_in_code, "the Windows hint no longer contains a winget command"

        section = " ".join(_canonical_section().split())
        for cmd in winget_in_code:
            assert cmd in section, (
                "README's Windows install command differs from the one "
                "ks_build.toolchain_hint() prints when a build fails:\n"
                f"  code: {cmd}"
            )

    def test_every_platform_hint_mentions_cmake(self):
        for name, hint in ks_build.TOOLCHAIN_HINTS.items():
            assert "cmake" in hint.lower(), f"the {name} hint never says how to get CMake"


class TestNoStalePlatformClaims:
    @pytest.mark.parametrize("doc", _docs(), ids=lambda p: str(p.name))
    def test_no_doc_claims_the_build_requires_macos(self, doc: Path):
        """The exact drift this file was written for.

        The model README said "Requires macOS with Command Line Tools" long after
        Linux worked, and still said it once Windows did. Statements about the
        *GPU* being Apple-only are true and untouched — this only rejects claims
        that the build or the model as a whole needs a Mac.
        """
        text = doc.read_text(encoding="utf-8")
        bad = re.findall(
            r"(?i)(requires?\s+(?:a\s+)?mac(?:OS)?\b[^\n.]*|mac(?:OS)?[- ]only\b[^\n.]*)",
            text,
        )
        # "The GPU path is Apple-only" and friends are correct; only macOS claims
        # are matched above, and the GPU exemption is spelled out explicitly.
        offenders = [m for m in bad if "gpu" not in m.lower() and "metal" not in m.lower()]
        assert not offenders, (
            f"{doc.relative_to(_REPO)} claims the build needs macOS: {offenders}. "
            "The model builds with MSVC, Clang and GCC, all tested on every push."
        )

    # Generous, but "near the top" is the property that matters: a reader who
    # skims the first screen must not come away thinking the model is Mac-only.
    _HEADER_LINES = 20

    @pytest.mark.parametrize("readme", _model_readmes(), ids=lambda p: p.parent.name)
    def test_each_model_readme_states_all_three_platforms_up_front(self, readme: Path):
        """A model README is read standalone on GitHub, with no surrounding context.

        Checked in the opening lines rather than anywhere in the file: an intro
        reading "runs on macOS and Linux" is not redeemed by a note about
        `ks_gpu.exe` fifty lines below, and a whole-file substring search cannot
        tell those apart. Discovered rather than listed, so the next compiled
        model inherits the rule.
        """
        header = "\n".join(readme.read_text(encoding="utf-8").splitlines()[: self._HEADER_LINES])
        for platform in ("Windows", "macOS", "Linux"):
            assert platform in header, (
                f"{readme.relative_to(_REPO)} does not mention {platform} in its first "
                f"{self._HEADER_LINES} lines; it is read standalone on GitHub, so what "
                "it runs on has to be visible without scrolling"
            )

    def test_the_discovery_finds_something(self):
        """Guard the guard: an empty model list makes every check above vacuous."""
        found = compiled_models()
        assert found, "no compiled models discovered — every doc check above is vacuous"
        assert any(p.name == "kinetic_segregation" for p in found), (
            f"expected kinetic_segregation among compiled models, saw {[p.name for p in found]}"
        )
