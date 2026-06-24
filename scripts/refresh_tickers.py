"""Refresh data/tasi_tickers.csv from SAHMK's /companies endpoint (full TASI universe).

Merges the live issuer list with the existing curated rows: curated rows keep their metadata
(name_ar, sector, company_type, shariah_flag) and stay FIRST (so the market scan's default
subset remains the well-known large-caps); new names are appended with SAHMK names, an inferred
company_type, and blank sector / unknown shariah (the live analysis fills sector at run time).

    python scripts/refresh_tickers.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from config.settings import get_config

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "tasi_tickers.csv"
FIELDS = ["ticker", "name_en", "name_ar", "sector", "company_type", "shariah_flag"]

# Tadawul sector code ranges / known sets for inferring company_type without a sector field.
BANK_CODES = {"1010", "1020", "1030", "1050", "1060", "1080", "1120", "1140", "1150", "1180"}


def infer_type(code: str) -> str:
    if code in BANK_CODES:
        return "bank"
    try:
        c = int(code)
    except ValueError:
        return "general"
    if 8000 <= c <= 8399:
        return "insurance"
    if 4330 <= c <= 4350:
        return "reit"
    return "general"


def fetch_all_companies(cfg) -> list[dict]:
    base, key = cfg.providers.saudi_exchange.base_url, cfg.sahmk_key
    if not key:
        raise SystemExit("SAHMK_API_KEY required to refresh tickers.")
    headers = {"X-API-Key": key, "Accept": "application/json"}
    out, offset, limit = [], 0, 200
    while True:
        r = requests.get(f"{base}/companies/?market=TASI&limit={limit}&offset={offset}",
                         headers=headers, timeout=30)
        r.raise_for_status()
        d = r.json()
        items = d.get("results") or []
        out.extend(items)
        total = d.get("total", len(out))
        offset += limit
        if offset >= total or not items:
            break
    return out


def load_existing() -> dict[str, dict]:
    if not CSV_PATH.exists():
        return {}
    with CSV_PATH.open(encoding="utf-8") as fh:
        return {row["ticker"].zfill(4): row for row in csv.DictReader(fh)}


def main():
    cfg = get_config()
    companies = fetch_all_companies(cfg)
    active = [c for c in companies if str(c.get("status", "active")).lower() == "active"]
    print(f"SAHMK returned {len(companies)} TASI names ({len(active)} active).")

    existing = load_existing()
    rows: list[dict] = []
    seen: set[str] = set()

    # 1) keep curated rows first, in their existing order
    for code, row in existing.items():
        rows.append({k: row.get(k, "") for k in FIELDS})
        seen.add(code)

    # 2) append new active names, sorted by ticker
    new = 0
    for c in sorted(active, key=lambda x: str(x.get("symbol", ""))):
        code = str(c.get("symbol", "")).strip().zfill(4)
        if not code or code in seen:
            continue
        rows.append({
            "ticker": code,
            "name_en": (c.get("name_en") or "").strip(),
            "name_ar": (c.get("name_ar") or "").strip(),
            "sector": "",  # filled live by the analyzer
            "company_type": infer_type(code),
            "shariah_flag": "unknown",
        })
        seen.add(code)
        new += 1

    with CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    types = {}
    for r in rows:
        types[r["company_type"]] = types.get(r["company_type"], 0) + 1
    print(f"Wrote {len(rows)} rows to {CSV_PATH} (+{new} new). Types: {types}")


if __name__ == "__main__":
    main()
