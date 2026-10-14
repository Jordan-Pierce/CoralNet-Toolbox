"""Assert a built wheel carries the package's non-Python data files.

Version 1.1.0 shipped with zero icons: commit 1efdb455 dropped setuptools_scm
from [build-system] requires as "unused", but its setuptools.file_finders entry
point was the only thing collecting git-tracked data files into the
distribution. MANIFEST.in named Layout/*.json and nothing else, so the 83
Icons/*.svg and the 17 Community/cfg/**/*.yaml silently stopped shipping. The
app does not crash without them -- get_icon_path() probes with is_file() and
returns None, community model lists just come up empty -- so nothing failed
loudly enough to catch it before release.

Run against a built wheel:

    python scripts/check_wheel_data.py dist/*.whl

Expectations come from the repository rather than a hardcoded count, so adding
an icon or a config needs no edit here.
"""

import subprocess
import sys
import zipfile
from pathlib import Path

PACKAGE = "coralnet_toolbox"

# Extensions that are part of the installed package. Anything tracked under
# PACKAGE with one of these must appear in the wheel.
DATA_SUFFIXES = {".svg", ".yaml", ".yml", ".json", ".png", ".qss", ".ui"}

REPO_ROOT = Path(__file__).resolve().parent.parent


def tracked_data_files():
    """Git-tracked data files under the package directory, as posix paths."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", PACKAGE],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        names = [n for n in out.split("\0") if n]
    except (OSError, subprocess.CalledProcessError):
        # No git (building from an unpacked sdist, say). Walk the tree instead.
        names = [
            p.relative_to(REPO_ROOT).as_posix()
            for p in (REPO_ROOT / PACKAGE).rglob("*")
            if p.is_file()
        ]

    return sorted(
        n for n in names
        if Path(n).suffix.lower() in DATA_SUFFIXES and "__pycache__" not in n
    )


def main(argv):
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} <wheel>", file=sys.stderr)
        return 2

    wheel = Path(argv[1])
    if not wheel.is_file():
        print(f"FAIL: no such wheel: {wheel}", file=sys.stderr)
        return 1

    expected = tracked_data_files()
    if not expected:
        # Nothing to check means the check itself is broken: a repo with no
        # icons would otherwise report success.
        print(f"FAIL: found no data files under {PACKAGE}/ to check for", file=sys.stderr)
        return 1

    with zipfile.ZipFile(wheel) as zf:
        present = set(zf.namelist())

    missing = [name for name in expected if name not in present]

    by_suffix = {}
    for name in expected:
        by_suffix[Path(name).suffix.lower()] = by_suffix.get(Path(name).suffix.lower(), 0) + 1
    summary = ", ".join(f"{count} {suffix}" for suffix, count in sorted(by_suffix.items()))

    if missing:
        print(f"FAIL: {wheel.name} is missing {len(missing)} of {len(expected)} data files ({summary})")
        print()
        print("setuptools_scm supplies the file finder that collects these; check that")
        print('[build-system] requires in pyproject.toml still lists "setuptools_scm>=8",')
        print("and that the build ran inside a git checkout or from a complete sdist.")
        print()
        for name in missing[:20]:
            print(f"  missing: {name}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        return 1

    print(f"OK: {wheel.name} carries all {len(expected)} data files ({summary})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
