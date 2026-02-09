"""
[ZINC22](https://cartblanche.docking.org/tranches/3d) molecules are downloaded as compressed
in sets and nested deep. Unpack and flatten.

Run with cli_args of the zinc "H" folder names.

This  is set up for the Curl download format; other formats use different packing.

Run with the  `--h_folders argument`
"""

import argparse
import shutil
import tarfile
from pathlib import Path

ARCHIVE_SUFFIXES = (".tgz", ".tar.gz", ".tzg")
BUBBLE_SUFFIXES = (".mol2", ".sdf")


def iter_archives_recursive(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if any(name.endswith(sfx) for sfx in ARCHIVE_SUFFIXES):
            yield p


def unique_dest_path(dest_dir: Path, filename: str) -> Path:
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    i = 1
    while True:
        c = dest_dir / f"{stem}.{i}{suffix}"
        if not c.exists():
            return c
        i += 1


def safe_unlink_empty_parents(path: Path, stop_at: Path):
    cur = path
    while True:
        if cur == stop_at:
            return
        try:
            cur.rmdir()
        except OSError:
            return
        cur = cur.parent


def extract_and_bubble_files(h_dir: Path, archive_path: Path) -> int:
    # Make temp dir unique to this archive path (including its parent) to avoid collisions.
    rel = archive_path.relative_to(h_dir)
    slug = "_".join(rel.parts).replace(".", "_")
    tmp_dir = h_dir / f".__extract_tmp__{slug}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    try:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(tmp_dir)

        for ext in BUBBLE_SUFFIXES:
            for f in tmp_dir.rglob(f"*{ext}"):
                dest = unique_dest_path(h_dir, f.name)
                shutil.move(str(f), str(dest))
                moved += 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return moved


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Recursively find Zinc archives under each provided Hxx folder, extract them, "
            "and move .mol2/.sdf files up to the Hxx folder."
        )
    )
    ap.add_argument(
        "--h_folders",
        nargs="+",
        help='One or more H folders (e.g. "H04 H05"). You can pass full paths.',
    )
    ap.add_argument(
        "--keep-archives-subfolders",
        action="store_true",
        help=(
            "Do not move the archive files themselves; only bubble molecule files. "
            "By default archives are left where they are (this flag just documents intent)."
        ),
    )
    args = ap.parse_args()

    total_archives = 0
    total_moved = 0

    for h in args.h_folders:
        h_dir = Path(h).expanduser().resolve()
        if not h_dir.is_dir():
            print(f"[skip] not a directory: {h_dir}")
            continue

        archives = list(iter_archives_recursive(h_dir))
        if not archives:
            print(f"[info] no archives found under {h_dir}")
            continue

        print(f"[info] {h_dir}: {len(archives)} archive(s) (recursive)")
        total_archives += len(archives)

        for a in archives:
            moved = extract_and_bubble_files(h_dir, a)
            total_moved += moved
            print(f"  {a.relative_to(h_dir)}: moved {moved} file(s)")

            # Optional cleanup: if you *want* to remove empty dirs left behind by archives
            # (not required for correctness), uncomment:
            # safe_unlink_empty_parents(a.parent, h_dir)

    print(f"[done] archives processed: {total_archives}; files moved: {total_moved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
