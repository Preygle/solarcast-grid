"""Download the Mendeley dataset y58jknpgs8 (v2, CC BY 4.0) and verify SHA-256 hashes.

Source: Mukherjee, Kalita, Kumar. "Electricity Demand, Solar and Wind Generation Data
(September 2021- June 2025) of India at 1-hour interval", Mendeley Data, V2,
doi:10.17632/y58jknpgs8.2

Usage:  python scripts/download_data.py
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "mendeley_y58jknpgs8"
API = "https://data.mendeley.com/public-api/datasets/y58jknpgs8/files?folder_id=root&version=2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(entry: dict) -> tuple[str, bool]:
    path = RAW / entry["filename"]
    expected = entry["content_details"]["sha256_hash"]
    if path.exists() and path.stat().st_size == entry["size"] and sha256(path) == expected:
        return entry["filename"], True
    with requests.get(entry["content_details"]["download_url"], stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return entry["filename"], sha256(path) == expected


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW / "_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(requests.get(API, timeout=60).text, encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ok = True
    with cf.ThreadPoolExecutor(6) as ex:
        for name, good in ex.map(fetch, manifest):
            print(f"{'OK ' if good else 'BAD'} {name}")
            ok &= good
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
