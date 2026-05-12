#!/usr/bin/env python3
"""YouTube metadata crawler with optional full-download mode (risk-ack gated)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class CandidateRecord:
    video_id: str
    url: str
    title: str
    description: str
    channel_id: str
    channel_title: str
    published_at: str
    license_filter_used: str
    review_status: str
    rights_status: str
    api_fetched_at: str
    source_query: str
    download_status: str = "not_requested"
    local_video_path: str = ""
    provenance_note: str = "metadata_only"


DEFAULT_QUERIES = [
    "hydraulic press crush",
    "hydraulic press experiment",
    "hydraulic press crushing objects",
    "hydraulic press glass",
    "hydraulic press metal",
    "hydraulic press rubber",
    "hydraulic press ceramic",
    "hydraulic press plastic",
    "hydraulic press wood",
    "hydraulic press fruit",
]


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def build_search_url(api_key: str, query: str, max_results: int) -> str:
    params = {
        "part": "snippet",
        "type": "video",
        "q": query,
        "maxResults": str(max_results),
    }
    encoded = urllib.parse.urlencode(params)
    return f"https://www.googleapis.com/youtube/v3/search?{encoded}&key={api_key}"


def fetch_search_page(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
        text = response.read().decode("utf-8")
    return json.loads(text)


def normalize_items(items: list[dict[str, Any]], query: str) -> list[CandidateRecord]:
    now = utc_now_iso()
    rows: list[CandidateRecord] = []
    for item in items:
        snippet = item.get("snippet", {})
        video_id = str(item.get("id", {}).get("videoId", "")).strip()
        if not video_id:
            continue
        rows.append(
            CandidateRecord(
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                title=str(snippet.get("title", "")),
                description=str(snippet.get("description", "")),
                channel_id=str(snippet.get("channelId", "")),
                channel_title=str(snippet.get("channelTitle", "")),
                published_at=str(snippet.get("publishedAt", "")),
                license_filter_used="any",
                review_status="needs_relevance_review",
                rights_status="needs_relevance_review",
                api_fetched_at=now,
                source_query=query,
            )
        )
    # keep query dependency explicit in logs even if not part of schema contract
    if not rows:
        sys.stderr.write(f"[warn] no videos for query={query!r}\n")
    return rows


def dedupe(records: list[CandidateRecord]) -> list[CandidateRecord]:
    by_id: dict[str, CandidateRecord] = {}
    for row in records:
        by_id[row.video_id] = row
    return list(by_id.values())


def write_json(path: Path, records: list[CandidateRecord]) -> None:
    payload = [asdict(r) for r in records]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, records: list[CandidateRecord]) -> None:
    fieldnames = list(asdict(records[0]).keys()) if records else list(CandidateRecord.__annotations__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(asdict(row))


def load_mock(path: Path) -> list[CandidateRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        msg = "Mock file must contain a list of normalized candidate rows."
        raise ValueError(msg)
    out: list[CandidateRecord] = []
    for row in data:
        merged = {
            "source_query": "mock_query",
            "download_status": "not_requested",
            "local_video_path": "",
            "provenance_note": "metadata_only",
            **row,
        }
        out.append(CandidateRecord(**merged))
    return out


def _write_risk_consent_log(path: Path) -> None:
    path.write_text(
        "full_auto_download enabled by explicit operator flag; user accepts policy/legal risk.\n",
        encoding="utf-8",
    )


def _download_records(
    records: list[CandidateRecord],
    *,
    download_dir: Path,
    simulate_download: bool,
) -> list[CandidateRecord]:
    download_dir.mkdir(parents=True, exist_ok=True)
    updated: list[CandidateRecord] = []
    for row in records:
        target = download_dir / f"{row.video_id}.mp4"
        status = "downloaded"
        note = "full_auto_download"
        try:
            if simulate_download:
                target.write_bytes(b"")
            else:
                # Best-effort direct fetch. For many YouTube URLs this may fail; failure is captured in status.
                urllib.request.urlretrieve(row.url, target)  # noqa: S310
        except Exception as exc:  # noqa: BLE001
            status = "download_failed"
            note = f"download_error:{type(exc).__name__}"
            target = Path("")
        updated.append(
            CandidateRecord(
                video_id=row.video_id,
                url=row.url,
                title=row.title,
                description=row.description,
                channel_id=row.channel_id,
                channel_title=row.channel_title,
                published_at=row.published_at,
                license_filter_used=row.license_filter_used,
                review_status=row.review_status,
                rights_status=row.rights_status,
                api_fetched_at=row.api_fetched_at,
                source_query=row.source_query,
                download_status=status,
                local_video_path=str(target) if str(target) else "",
                provenance_note=note,
            )
        )
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="YouTube candidate crawler with optional full download mode")
    parser.add_argument("--out-dir", type=Path, default=Path("data/candidates"))
    parser.add_argument("--download-dir", type=Path, default=Path("data/raw_video"))
    parser.add_argument("--max-results", type=int, default=25)
    parser.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES)
    parser.add_argument("--mock-json", type=Path, default=None, help="Use local normalized rows instead of API calls")
    parser.add_argument("--enable-full-download", action="store_true")
    parser.add_argument("--allow-risky-download", action="store_true", help="Required with --enable-full-download")
    parser.add_argument("--simulate-download", action="store_true", help="Create placeholder files instead of network fetch")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mock_json is not None:
        rows = dedupe(load_mock(args.mock_json))
    else:
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            sys.stderr.write("YOUTUBE_API_KEY is required unless --mock-json is provided.\n")
            return 2
        rows_all: list[CandidateRecord] = []
        for query in args.queries:
            url = build_search_url(api_key, query, args.max_results)
            payload = fetch_search_page(url)
            items = payload.get("items", [])
            if not isinstance(items, list):
                continue
            rows_all.extend(normalize_items(items, query))
        rows = dedupe(rows_all)

    if args.enable_full_download:
        if not args.allow_risky_download:
            sys.stderr.write(
                "Full download mode requested. Re-run with --allow-risky-download to acknowledge risk.\n"
            )
            return 3
        consent_log = args.out_dir / "full_download_consent.log"
        _write_risk_consent_log(consent_log)
        rows = _download_records(
            rows,
            download_dir=args.download_dir,
            simulate_download=args.simulate_download,
        )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = args.out_dir / f"candidates_{timestamp}.json"
    csv_path = args.out_dir / f"candidates_{timestamp}.csv"
    write_json(json_path, rows)
    write_csv(csv_path, rows)
    print(f"wrote {len(rows)} candidates")
    print(f"json: {json_path}")
    print(f"csv : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
