#!/usr/bin/env python3
"""Download monthly 1-minute kline archives from the public Binance data mirror.

Files come from https://data.binance.vision (no account or API key needed) and are
saved unchanged as ``data/raw/<SYMBOL>-<interval>-<YYYY>-<MM>.zip``. Each archive is
checked against the SHA-256 checksum that Binance publishes next to it.

Example:
    uv run python scripts/download_data.py --year 2024
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "raw"
USER_AGENT = "btc-mean-reversion-research/0.1"


def archive_url(symbol: str, interval: str, year: int, month: int) -> str:
    """Return the URL of one monthly kline archive."""
    name = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    return f"{BASE_URL}/{symbol}/{interval}/{name}"


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _expected_sha256(url: str) -> str | None:
    """Read the published checksum (format: '<sha256>  <filename>'); None if unavailable."""
    try:
        text = _fetch(url + ".CHECKSUM").decode("ascii").strip()
    except urllib.error.URLError:
        return None
    return text.split()[0].lower() if text else None


def download_month(
    symbol: str, interval: str, year: int, month: int, out_dir: Path, force: bool = False
) -> Path:
    """Download and verify one monthly archive; skip it if already present."""
    url = archive_url(symbol, interval, year, month)
    target = out_dir / url.rsplit("/", 1)[-1]
    if target.exists() and not force:
        print(f"exists   {target.name}")
        return target

    payload = _fetch(url)
    expected = _expected_sha256(url)
    actual = hashlib.sha256(payload).hexdigest()
    if expected is not None and expected != actual:
        raise RuntimeError(f"Checksum mismatch for {target.name}: {actual} != {expected}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".zip.part")
    tmp.write_bytes(payload)
    tmp.replace(target)
    status = "verified" if expected else "no checksum"
    print(f"saved    {target.name}  ({len(payload) / 1e6:.1f} MB, {status})")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--months", type=int, nargs="+", default=list(range(1, 13)), help="Months 1-12."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures = []
    for month in args.months:
        try:
            download_month(args.symbol, args.interval, args.year, month, args.out_dir, args.force)
        except (urllib.error.URLError, RuntimeError) as exc:
            print(f"FAILED   {args.year}-{month:02d}: {exc}", file=sys.stderr)
            failures.append(month)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
