"""Generate scenario forecasts for a new Russian-language news item."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from config import COMPANIES, HORIZONS, MODEL_DIR, RESULTS_DIR, SECTOR_KEYWORDS
from src.features import event_type, sentiment_label, sentiment_score
from src.train_model import FEATURE_COLUMNS


MSK = ZoneInfo("Europe/Moscow")


def identify_companies(text: str, explicit_tickers: list[str] | None = None) -> list[tuple[str, str]]:
    low = text.lower()
    if explicit_tickers:
        selected = []
        seen = set()
        for ticker in explicit_tickers:
            ticker = ticker.upper().strip()
            if ticker not in COMPANIES:
                raise ValueError(f"Unknown ticker: {ticker}")
            if ticker not in seen:
                selected.append((ticker, "explicit"))
                seen.add(ticker)
        return selected

    direct = []
    for ticker, company in COMPANIES.items():
        if any(alias.lower() in low for alias in company["aliases"]):
            direct.append((ticker, "company alias in text"))
    if direct:
        return direct

    sector_matches = []
    for sector, keywords in SECTOR_KEYWORDS.items():
        score = sum(keyword in low for keyword in keywords)
        if score:
            sector_matches.append((score, sector))
    if not sector_matches:
        raise ValueError("No company or sector identified. Pass --tickers, for example --tickers SBER,VTBR")
    best_sector = max(sector_matches)[1]
    return [(ticker, "sector keyword fallback") for ticker, item in COMPANIES.items() if item["sector"] == best_sector]


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.now(MSK)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=MSK)
    return parsed.astimezone(MSK)


def predict(
    text: str,
    source: str = "user_input",
    published_at: str | None = None,
    tickers: list[str] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    bundle = joblib.load(MODEL_DIR / "model_bundle.joblib")
    published = _parse_timestamp(published_at)
    score = sentiment_score(text)
    kind = event_type(text)
    selected = identify_companies(text, tickers)

    feature_rows = []
    metadata = []
    for ticker, match_reason in selected:
        company = COMPANIES[ticker]
        sentiment = sentiment_label(score)
        feature_rows.append(
            {
                "title": text,
                "ticker": ticker,
                "sector": company["sector"],
                "source_name": source,
                "event_type_rule": kind,
                "sentiment_score_rule": score,
                "publication_hour_msk": published.hour,
                "publication_weekday": published.weekday(),
                "publication_is_weekend": int(published.weekday() >= 5),
                "publication_month": published.month,
            }
        )
        metadata.append(
            {
                "company": company["name"],
                "ticker": ticker,
                "sector": company["sector"],
                "match_reason": match_reason,
                "event_type": kind,
                "sentiment": sentiment,
            }
        )
    features = pd.DataFrame(feature_rows)[FEATURE_COLUMNS]
    result = pd.DataFrame(metadata)

    for horizon in HORIZONS:
        model_info = bundle["horizons"][horizon]
        raw_pred = model_info["regressor"].predict(features)
        pred = np.clip(raw_pred, model_info["clip_q01"], model_info["clip_q99"])
        probability = model_info["classifier"].predict_proba(features)[:, 1]
        low = pred + model_info["residual_q10"]
        high = pred + model_info["residual_q90"]
        result[f"forecast_{horizon}d_pct"] = np.round(pred * 100, 3)
        result[f"interval80_low_{horizon}d_pct"] = np.round(low * 100, 3)
        result[f"interval80_high_{horizon}d_pct"] = np.round(high * 100, 3)
        result[f"positive_probability_{horizon}d"] = np.round(probability, 3)

    result.insert(0, "published_at_msk", published.isoformat())
    result.insert(0, "source", source)
    result.insert(0, "news_text", text)
    result["forecast_type"] = "abnormal return vs IMOEX; market-neutral price-change scenario"
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        result.to_csv(RESULTS_DIR / "latest_prediction.csv", index=False, encoding="utf-8-sig")
        (RESULTS_DIR / "latest_prediction.json").write_text(
            json.dumps(result.to_dict("records"), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast stock reaction to new news")
    parser.add_argument("--text", required=True, help="Headline or short news text")
    parser.add_argument("--source", default="user_input")
    parser.add_argument("--published-at", help="ISO timestamp; Moscow time if no timezone is supplied")
    parser.add_argument("--tickers", help="Comma-separated MOEX tickers; optional when company is named in text")
    args = parser.parse_args()
    tickers = args.tickers.split(",") if args.tickers else None
    output = predict(args.text, args.source, args.published_at, tickers)
    display_columns = ["company", "ticker"] + [f"forecast_{h}d_pct" for h in HORIZONS]
    print(output[display_columns].to_string(index=False))
    print("\nSaved detailed forecast to results/latest_prediction.csv and .json")


if __name__ == "__main__":
    main()
