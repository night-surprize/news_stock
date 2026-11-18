import unittest

import joblib
import pandas as pd

from config import DATA_DIR, MODEL_DIR
from src.features import event_type, is_reaction_reporting, sentiment_score
from src.predict_news import identify_companies, predict


class FeatureTests(unittest.TestCase):
    def test_rule_based_features(self) -> None:
        self.assertEqual(sentiment_score("Рост прибыли"), 2)
        self.assertEqual(sentiment_score("Санкции и убыток"), -2)
        self.assertEqual(event_type("Совет директоров рекомендовал дивиденды"), "dividends")
        self.assertTrue(is_reaction_reporting("Акции компании выросли после отчёта"))

    def test_company_selection(self) -> None:
        self.assertEqual(identify_companies("Сбербанк увеличил прибыль"), [("SBER", "company alias in text")])
        self.assertEqual(
            identify_companies("Новость", ["sber", "SBER"]),
            [("SBER", "explicit")],
        )


class ProjectSmokeTests(unittest.TestCase):
    def test_prepared_artifacts(self) -> None:
        required = [
            DATA_DIR / "news_raw.csv",
            DATA_DIR / "moex_prices_daily.csv",
            DATA_DIR / "modeling_dataset.csv",
            MODEL_DIR / "model_bundle.joblib",
        ]
        for path in required:
            self.assertTrue(path.is_file(), path)
            self.assertGreater(path.stat().st_size, 0, path)

        news = pd.read_csv(DATA_DIR / "news_raw.csv")
        modeling = pd.read_csv(DATA_DIR / "modeling_dataset.csv")
        self.assertGreaterEqual(len(news), 500)
        self.assertGreaterEqual(len(modeling), 400)
        self.assertTrue(set(modeling["ticker"]).issubset(news["ticker"]))

        bundle = joblib.load(MODEL_DIR / "model_bundle.joblib")
        self.assertEqual(set(bundle["horizons"]), {1, 3, 5})

    def test_prediction(self) -> None:
        forecast = predict(
            "Сбербанк сообщил о росте прибыли",
            published_at="2026-11-05T12:30:00+03:00",
            tickers=["SBER"],
            save=False,
        )
        self.assertEqual(len(forecast), 1)
        self.assertTrue(forecast.filter(regex=r"forecast_\dd_pct").notna().all().all())


if __name__ == "__main__":
    unittest.main()
