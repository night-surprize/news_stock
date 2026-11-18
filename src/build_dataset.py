"""Align news events with trading sessions and compute event-study targets."""

from __future__ import annotations

import bisect
import json
from datetime import datetime, time, timezone

import numpy as np
import pandas as pd

from config import (
    BENCHMARK,
    DATA_DIR,
    HORIZONS,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
)
from src.features import event_type, is_reaction_reporting, sentiment_label, sentiment_score


def _event_session_index(pub_msk: pd.Timestamp, trading_dates: list[pd.Timestamp]) -> int | None:
    day = pub_msk.normalize().tz_localize(None)
    naive_dates = [d.tz_localize(None) if d.tzinfo else d for d in trading_dates]
    left = bisect.bisect_left(naive_dates, day)
    is_trading_day = left < len(naive_dates) and naive_dates[left] == day
    before_close = (pub_msk.hour, pub_msk.minute) <= (MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)
    if is_trading_day and before_close:
        return left
    return bisect.bisect_right(naive_dates, day)


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    news = pd.read_csv(DATA_DIR / "news_raw.csv")
    prices = pd.read_csv(DATA_DIR / "moex_prices_daily.csv")
    news["published_at_msk"] = pd.to_datetime(news["published_at_msk"], utc=True).dt.tz_convert("Europe/Moscow")
    news["sentiment_score_rule"] = news["title"].map(sentiment_score)
    news["sentiment_rule"] = news["sentiment_score_rule"].map(sentiment_label)
    news["event_type_rule"] = news["title"].map(event_type)
    news["reaction_reporting_flag"] = news["title"].map(is_reaction_reporting)
    prices["date"] = pd.to_datetime(prices["date"])

    price_tables: dict[str, pd.DataFrame] = {}
    for ticker, group in prices.groupby("ticker"):
        table = group.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
        price_tables[ticker] = table
    benchmark = price_tables[BENCHMARK]["close"].astype(float)

    output = []
    for row in news.to_dict("records"):
        ticker = row["ticker"]
        table = price_tables.get(ticker)
        if table is None or table.empty:
            continue
        dates = list(table.index)
        pub = pd.Timestamp(row["published_at_msk"])
        event_idx = _event_session_index(pub, dates)
        result = dict(row)
        result["publication_date_msk"] = pub.date().isoformat()
        result["publication_hour_msk"] = pub.hour
        result["publication_weekday"] = pub.weekday()
        result["publication_is_weekend"] = int(pub.weekday() >= 5)
        result["publication_year"] = pub.year
        result["publication_month"] = pub.month
        result["alignment_status"] = "ok"

        if event_idx is None or event_idx <= 0 or event_idx + max(HORIZONS) - 1 >= len(dates):
            result["alignment_status"] = "insufficient_price_window"
            output.append(result)
            continue

        baseline_date = dates[event_idx - 1]
        event_date = dates[event_idx]
        result["baseline_date"] = baseline_date.date().isoformat()
        result["event_session_date"] = event_date.date().isoformat()
        baseline_close = float(table.loc[baseline_date, "close"])
        result["baseline_close"] = baseline_close

        if baseline_date not in benchmark.index:
            result["alignment_status"] = "benchmark_missing_baseline"
            output.append(result)
            continue
        benchmark_base = float(benchmark.loc[baseline_date])

        for horizon in HORIZONS:
            target_date = dates[event_idx + horizon - 1]
            result[f"target_date_{horizon}d"] = target_date.date().isoformat()
            result[f"close_{horizon}d"] = float(table.loc[target_date, "close"])
            stock_return = float(table.loc[target_date, "close"] / baseline_close - 1.0)
            result[f"stock_return_{horizon}d"] = stock_return
            if target_date in benchmark.index:
                market_return = float(benchmark.loc[target_date] / benchmark_base - 1.0)
                result[f"market_return_{horizon}d"] = market_return
                result[f"abnormal_return_{horizon}d"] = stock_return - market_return
                result[f"direction_{horizon}d"] = int(stock_return - market_return > 0)
            else:
                result[f"market_return_{horizon}d"] = np.nan
                result[f"abnormal_return_{horizon}d"] = np.nan
                result[f"direction_{horizon}d"] = np.nan
        output.append(result)

    all_events = pd.DataFrame(output).sort_values(["published_at_msk", "ticker"])
    all_events.to_csv(DATA_DIR / "events_with_returns_all.csv", index=False, encoding="utf-8-sig")

    required = [f"abnormal_return_{h}d" for h in HORIZONS]
    modeling = all_events.loc[all_events["alignment_status"] == "ok"].dropna(subset=required).copy()
    modeling["extreme_return_flag"] = False
    for horizon in HORIZONS:
        modeling["extreme_return_flag"] |= modeling[f"stock_return_{horizon}d"].abs() > 0.50
        modeling["extreme_return_flag"] |= modeling[f"market_return_{horizon}d"].abs() > 0.30
    before_filters = len(modeling)
    modeling = modeling.loc[~modeling["reaction_reporting_flag"].astype(bool)]
    modeling = modeling.loc[~modeling["extreme_return_flag"]]
    modeling = modeling.drop_duplicates(subset=["ticker", "title", "event_session_date"], keep="first")
    modeling.to_csv(DATA_DIR / "modeling_dataset.csv", index=False, encoding="utf-8-sig")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "news_rows": int(len(news)),
        "aligned_rows_before_filters": int(before_filters),
        "modeling_rows": int(len(modeling)),
        "removed_reaction_reporting_or_extreme_or_duplicate": int(before_filters - len(modeling)),
        "alignment_status": all_events["alignment_status"].value_counts(dropna=False).to_dict(),
        "modeling_rows_by_sector": modeling["sector"].value_counts().to_dict(),
        "modeling_rows_by_ticker": modeling["ticker"].value_counts().to_dict(),
        "method": {
            "event_session": "same trading day if RSS timestamp is no later than 18:45 MSK; otherwise next trading day",
            "baseline": "previous trading-day close",
            "targets": "close-to-close total return minus IMOEX return over 1, 3 and 5 trading sessions",
            "leakage_filter": "headlines explicitly reporting an already observed stock-price move are excluded",
        },
    }
    (DATA_DIR / "dataset_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return all_events, modeling


if __name__ == "__main__":
    all_events, modeling = build_dataset()
    print(f"Aligned: {len(all_events)}; modeling sample: {len(modeling)}")
