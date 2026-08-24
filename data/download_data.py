#!/usr/bin/env python3
"""Download the published H5 measurement files and verify sha256 against manifest.json."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ZENODO_RECORD_URL = "TODO: fill in after upload"

DATA_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = DATA_DIR / "manifest.json"
CHUNK = 1024 * 1024


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _progress(prefix: str, got: int, total: int | None) -> None:
    if total:
        pct = 100.0 * got / total
        print(f"\r{prefix} {got / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.1f}%)", end="", flush=True)
    else:
        print(f"\r{prefix} {got / 1e6:.1f} MB", end="", flush=True)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as resp:
            total = resp.headers.get("Content-Length")
            total_n = int(total) if total else None
            got = 0
            with tmp.open("wb") as out:
                while True:
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
                    got += len(block)
                    _progress(f"Downloading {dest.name}", got, total_n)
        print()
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download published H5 measurement files into data/ and verify sha256."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if a file is already present and valid.",
    )
    args = parser.parse_args()

    if ZENODO_RECORD_URL.startswith("TODO"):
        print(
            "data/download_data.py: ZENODO_RECORD_URL is still a placeholder.\n"
            "Upload the two H5 files to Zenodo, mint a DOI, then set ZENODO_RECORD_URL\n"
            "in this file (and the matching field in data/manifest.json).\n"
            "Until then, place measurement_motor_pcc.h5 and measurement_raf.h5 in data/\n"
            "manually. See data/README.md.",
            file=sys.stderr,
        )
        return 1

    manifest = _load_manifest()
    files = manifest["files"]
    failed = False
    for entry in files:
        name = entry["filename"]
        dest = DATA_DIR / name
        expected = entry["sha256"]
        if dest.exists() and not args.force:
            digest = _sha256(dest)
            if digest == expected:
                print(f"{name}: already present and sha256 matches.")
                continue
            print(f"{name}: present but sha256 mismatch; re-downloading.")
        url = ZENODO_RECORD_URL.rstrip("/") + "/files/" + name
        try:
            _download(url, dest)
        except urllib.error.URLError as exc:
            print(f"Failed to download {name}: {exc}", file=sys.stderr)
            failed = True
            continue
        digest = _sha256(dest)
        if digest != expected:
            print(
                f"{name}: sha256 mismatch\n  expected {expected}\n  got      {digest}",
                file=sys.stderr,
            )
            failed = True
        else:
            print(f"{name}: downloaded and verified.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
