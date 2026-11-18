"""Collect a reproducible headline dataset through Google News RSS indexing.

The RSS feed is used as an index over the required outlets. Each row retains
the outlet name/homepage and the Google News article link. Publication time is
the timestamp exposed by RSS and is explicitly marked as potentially normalized.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd

from config import CACHE_DIR, COMPANIES, DATA_DIR, END_DATE, NEWS_SOURCES, START_DATE
from src.features import event_type, is_reaction_reporting, sentiment_label, sentiment_score


RSS_ENDPOINT = "https://news.google.com/rss/search"
MSK = timezone(timedelta(hours=3))
USER_AGENT = "Mozilla/5.0 (compatible; educational-research/1.0)"


def _year_windows(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    windows = []
    for year in range(start.year, end.year + 1):
        left = max(start, pd.Timestamp(year=year, month=1, day=1))
        right_inclusive = min(end, pd.Timestamp(year=year, month=12, day=31))
        windows.append((left.strftime("%Y-%m-%d"), (right_inclusive + pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
    return windows


def _request_bytes(url: str, cache_path: Path, retries: int = 2) -> bytes:
    if cache_path.exists() and cache_path.stat().st_size > 100:
        return cache_path.read_bytes()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    error = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            if len(payload) < 100:
                raise ValueError("RSS response is unexpectedly small")
            cache_path.write_bytes(payload)
            return payload
        except Exception as exc:
            error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Unable to fetch RSS after retries: {error}")


def _clean_title(title: str, source_name: str) -> str:
    suffix = f" - {source_name}".lower()
    cleaned = " ".join((title or "").split())
    if cleaned.lower().endswith(suffix):
        cleaned = cleaned[: -len(suffix)].strip()
    return cleaned


def _parse_items(payload: bytes) -> list[dict]:
    root = ET.fromstring(payload)
    items = []
    for node in root.findall(".//item"):
        source_node = node.find("source")
        source_name = source_node.text.strip() if source_node is not None and source_node.text else "unknown"
        source_url = source_node.attrib.get("url", "") if source_node is not None else ""
        published_raw = node.findtext("pubDate") or ""
        try:
            published = parsedate_to_datetime(published_raw)
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            published = published.astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError):
            continue
        title_raw = node.findtext("title") or ""
        items.append(
            {
                "title": _clean_title(title_raw, source_name),
                "title_raw": title_raw,
                "published_at_utc": published.isoformat(),
                "published_at_msk": published.astimezone(MSK).isoformat(),
                "url": node.findtext("link") or "",
                "guid": node.findtext("guid") or "",
                "source_name": source_name,
                "source_homepage": source_url,
            }
        )
    return items


def _even_sample(items: list[dict], n: int) -> list[dict]:
    if len(items) <= n:
        return items
    ordered = sorted(items, key=lambda x: x["published_at_utc"])
    if n == 1:
        return [ordered[len(ordered) // 2]]
    indices = sorted({round(i * (len(ordered) - 1) / (n - 1)) for i in range(n)})
    return [ordered[i] for i in indices]


def _fetch_one(task: tuple[str, dict, str, dict, str, str], per_query: int) -> tuple[list[dict], dict]:
    ticker, company, source_key, source, after, before = task
    query = f'{company["query"]} {source["query"]} after:{after} before:{before}'
    params = {"q": query, "hl": "ru", "gl": "RU", "ceid": "RU:ru"}
    url = RSS_ENDPOINT + "?" + urllib.parse.urlencode(params)
    cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    cache_path = CACHE_DIR / "news_rss" / f"{cache_key}.xml"
    try:
        payload = _request_bytes(url, cache_path)
        parsed = _parse_items(payload)
        left = pd.Timestamp(after, tz="UTC")
        right = pd.Timestamp(before, tz="UTC")
        parsed = [
            item for item in parsed
            if left <= pd.Timestamp(item["published_at_utc"]) < right
        ]
        chosen = _even_sample(parsed, per_query)
        rows = []
        for item in chosen:
            score = sentiment_score(item["title"])
            event_id = hashlib.sha1(f'{ticker}|{item["url"]}|{item["published_at_utc"]}'.encode("utf-8")).hexdigest()[:16]
            rows.append(
                {
                    "event_id": event_id,
                    "ticker": ticker,
                    "company": company["name"],
                    "sector": company["sector"],
                    "title": item["title"],
                    "published_at_utc": item["published_at_utc"],
                    "published_at_msk": item["published_at_msk"],
                    "source_name": item["source_name"],
                    "source_homepage": item["source_homepage"],
                    "source_group": source["group"],
                    "requested_source": source["label"],
                    "url": item["url"],
                    "timestamp_quality": "Google News RSS timestamp; may be normalized by the index",
                    "content_scope": "headline",
                    "query_used": query,
                    "sentiment_score_rule": score,
                    "sentiment_rule": sentiment_label(score),
                    "event_type_rule": event_type(item["title"]),
                    "reaction_reporting_flag": is_reaction_reporting(item["title"]),
                    "collection_time_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
        return rows, {
            "ticker": ticker,
            "source": source_key,
            "after": after,
            "before": before,
            "rss_items": len(parsed),
            "selected": len(rows),
            "status": "ok",
        }
    except Exception as exc:
        return [], {
            "ticker": ticker,
            "source": source_key,
            "after": after,
            "before": before,
            "rss_items": 0,
            "selected": 0,
            "status": "error",
            "error": repr(exc),
        }


def collect_news(per_query: int = 2, workers: int = 20) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for ticker, company in COMPANIES.items():
        for source_key, source in NEWS_SOURCES.items():
            for after, before in _year_windows(START_DATE, END_DATE):
                tasks.append((ticker, company, source_key, source, after, before))

    rows: list[dict] = []
    log: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_fetch_one, task, per_query) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result_rows, result_log = future.result()
            rows.extend(result_rows)
            log.append(result_log)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No news records were collected")
    frame["published_at_utc"] = pd.to_datetime(frame["published_at_utc"], utc=True)
    frame = frame.sort_values(["published_at_utc", "ticker", "source_name"])
    frame = frame.drop_duplicates(subset=["ticker", "title", "published_at_utc"], keep="first")
    frame.to_csv(DATA_DIR / "news_raw.csv", index=False, encoding="utf-8-sig")

    quality = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "tasks": len(tasks),
        "successful_tasks": sum(item["status"] == "ok" for item in log),
        "failed_tasks": sum(item["status"] != "ok" for item in log),
        "rows_after_deduplication": int(len(frame)),
        "rows_by_requested_source": frame["requested_source"].value_counts().to_dict(),
        "rows_by_sector": frame["sector"].value_counts().to_dict(),
        "limitations": [
            "Dataset contains headlines, not full copyrighted article text.",
            "RSS publication timestamps can be normalized and are not guaranteed to equal the outlet page timestamp.",
            "Google News ranking means the sample is not a complete census of all published articles.",
        ],
        "task_log": sorted(log, key=lambda x: (x["ticker"], x["source"], x["after"])),
    }
    (DATA_DIR / "news_collection_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-query", type=int, default=2, help="Evenly sampled headlines per company/source/year query")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    frame = collect_news(per_query=args.per_query, workers=args.workers)
    print(f"Saved {len(frame)} news-ticker rows to {DATA_DIR / 'news_raw.csv'}")


if __name__ == "__main__":
    main()
