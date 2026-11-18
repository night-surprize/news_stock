"""Download daily OHLCV candles from the Moscow Exchange ISS API."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from config import BENCHMARK, CACHE_DIR, COMPANIES, DATA_DIR, END_DATE, START_DATE


USER_AGENT = "Mozilla/5.0 (compatible; educational-research/1.0)"


def _endpoint(ticker: str) -> str:
    if ticker == BENCHMARK:
        return f"https://iss.moex.com/iss/engines/stock/markets/index/securities/{ticker}/candles.json"
    return f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json"


def _get_json(url: str, retries: int = 2) -> dict:
    cache_key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    cache_path = CACHE_DIR / "moex" / f"{cache_key}.json"
    if cache_path.exists() and cache_path.stat().st_size > 100:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    error = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            data = json.loads(payload)
            cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        except Exception as exc:
            error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"MOEX request failed: {error}")


def fetch_ticker(ticker: str) -> pd.DataFrame:
    rows = []
    columns = None
    start = 0
    while True:
        params = {
            "from": START_DATE,
            "till": END_DATE,
            "interval": 24,
            "iss.meta": "off",
            "start": start,
        }
        url = _endpoint(ticker) + "?" + urllib.parse.urlencode(params)
        block = _get_json(url).get("candles", {})
        data = block.get("data", [])
        columns = block.get("columns", columns)
        if not data:
            break
        rows.extend(data)
        if len(data) < 500:
            break
        start += len(data)
    if not rows or not columns:
        raise RuntimeError(f"No candles returned for {ticker}")
    frame = pd.DataFrame(rows, columns=columns)
    frame.insert(0, "ticker", ticker)
    frame["date"] = pd.to_datetime(frame["begin"]).dt.date.astype(str)
    frame["source"] = "MOEX ISS"
    return frame[["ticker", "date", "open", "close", "high", "low", "value", "volume", "begin", "end", "source"]]


def collect_prices(workers: int = 12) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tickers = list(COMPANIES) + [BENCHMARK]
    frames = []
    errors = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(fetch_ticker, ticker): ticker for ticker in tickers}
        for future in concurrent.futures.as_completed(future_map):
            ticker = future_map[future]
            try:
                frames.append(future.result())
            except Exception as exc:
                errors[ticker] = repr(exc)
    if errors:
        raise RuntimeError(f"Price collection failed for: {errors}")
    prices = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    prices = prices.drop_duplicates(subset=["ticker", "date"], keep="last")
    prices.to_csv(DATA_DIR / "moex_prices_daily.csv", index=False, encoding="utf-8-sig")

    company_rows = []
    for ticker, item in COMPANIES.items():
        company_rows.append(
            {
                "ticker": ticker,
                "exchange": "Moscow Exchange",
                "company": item["name"],
                "sector": item["sector"],
                "business": item["business"],
                "geography": item["geography"],
                "aliases": "; ".join(item["aliases"]),
                "first_price_date": prices.loc[prices["ticker"] == ticker, "date"].min(),
                "last_price_date": prices.loc[prices["ticker"] == ticker, "date"].max(),
                "price_rows": int((prices["ticker"] == ticker).sum()),
            }
        )
    pd.DataFrame(company_rows).to_csv(DATA_DIR / "companies.csv", index=False, encoding="utf-8-sig")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Moscow Exchange ISS API",
        "api_documentation": "https://www.moex.com/a8531",
        "period": [START_DATE, END_DATE],
        "benchmark": BENCHMARK,
        "rows": int(len(prices)),
        "rows_by_ticker": prices.groupby("ticker").size().astype(int).to_dict(),
    }
    (DATA_DIR / "price_collection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return prices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    prices = collect_prices(workers=args.workers)
    print(f"Saved {len(prices)} price rows to {DATA_DIR / 'moex_prices_daily.csv'}")


if __name__ == "__main__":
    main()
