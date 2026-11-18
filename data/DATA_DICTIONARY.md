# Словарь данных

## companies.csv

- `ticker` - тикер MOEX;
- `company` - название эмитента;
- `sector` - одна из четырёх отраслевых групп;
- `business`, `geography` - описание бизнеса и географии;
- `first_price_date`, `last_price_date`, `price_rows` - покрытие котировок.

## news_raw.csv

- `event_id` - идентификатор пары "новость - тикер";
- `ticker`, `company`, `sector` - объект анализа;
- `title` - заголовок;
- `published_at_utc`, `published_at_msk` - время из RSS;
- `source_name`, `source_homepage` - исходное СМИ;
- `requested_source`, `source_group` - группа поискового запроса;
- `url` - ссылка Google News;
- `timestamp_quality` - предупреждение о качестве времени;
- `sentiment_*`, `event_type_rule` - правиловые признаки.

## moex_prices_daily.csv

- `ticker`, `date` - инструмент и дата;
- `open`, `close`, `high`, `low` - дневные цены;
- `value`, `volume` - оборот и количество;
- `begin`, `end` - границы свечи MOEX ISS.

## events_with_returns_all.csv

Все новости после попытки сопоставления с котировками. Поле `alignment_status` показывает, почему событие вошло или не вошло в расчёт.

## modeling_dataset.csv

Очищенная выборка для модели. Для каждого горизонта `h`:

- `stock_return_hd` - доходность акции;
- `market_return_hd` - доходность IMOEX;
- `abnormal_return_hd` - разность двух доходностей;
- `direction_hd` - знак аномальной доходности.
