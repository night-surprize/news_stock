"""Run the complete reproducible pipeline from the project root."""

from src.build_dataset import build_dataset
from src.collect_news import collect_news
from src.collect_prices import collect_prices
from src.train_model import train_models


def main() -> None:
    print("1/4 Collecting news...")
    news = collect_news()
    print(f"  {len(news)} news-ticker rows")
    print("2/4 Collecting MOEX prices...")
    prices = collect_prices()
    print(f"  {len(prices)} price rows")
    print("3/4 Building event-study dataset...")
    _, modeling = build_dataset()
    print(f"  {len(modeling)} modeling rows")
    print("4/4 Training and evaluating models...")
    train_models()
    print("Done. See data/, models/ and results/.")


if __name__ == "__main__":
    main()
