"""Deterministic, inspectable features for Russian financial headlines."""

import re


POSITIVE = {
    "рост", "вырос", "увелич", "прибыл", "дивиденд", "контракт", "запуст",
    "рекорд", "повыс", "одобр", "побед", "расшир", "инвестиц", "восстанов",
    "снизил долг", "сильн", "успеш", "экспорт вырос", "добыча выросла",
}
NEGATIVE = {
    "паден", "упал", "сниз", "убыт", "санкц", "авари", "пожар", "иск",
    "штраф", "атак", "дефолт", "огранич", "запрет", "сократ", "приостанов",
    "отказ", "риск", "проблем", "отозвал", "обруш", "утечк", "банкрот",
}

EVENT_PATTERNS = [
    ("dividends", r"дивиденд|выплат.*акционер"),
    ("secondary_offering", r"продаж.*(?:пакет|дол[ию]|акци)|госпакет|spo|free.?float|допэмисс"),
    ("financial_results", r"прибыл|убыт|выручк|ebitda|мсфо|рсбу|отчетност"),
    ("sanctions", r"санкц|ограничени.*экспорт|черн.*спис|эмбарго"),
    ("regulation_rates", r"ключев.*ставк|банк россии|центробанк|налог|пошлин|регулирован"),
    ("contracts_projects", r"контракт|сделк|проект|инвестиц|запуск|месторожд|строительств"),
    ("incidents", r"авари|пожар|атак|взрыв|останов|утечк|чрезвычайн"),
    ("commodities_macro", r"нефт|газ|удобрени|зерн|инфляц|курс рубл|опек|фрахт"),
    ("management", r"назнач|директор|правлени|руковод|менеджмент|акционер"),
]

REACTION_REPORTING_PATTERN = re.compile(
    r"(?:акци|бумаг).{0,30}(?:вырос|упал|подорожал|снизил|обвалил|прибавил)|"
    r"(?:выросли|упали|подорожали|снизились).{0,20}(?:акци|бумаг)",
    flags=re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def sentiment_score(text: str) -> int:
    low = normalize_text(text).lower()
    return sum(token in low for token in POSITIVE) - sum(token in low for token in NEGATIVE)


def sentiment_label(score: int) -> str:
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def event_type(text: str) -> str:
    low = normalize_text(text).lower()
    for label, pattern in EVENT_PATTERNS:
        if re.search(pattern, low):
            return label
    return "other"


def is_reaction_reporting(text: str) -> bool:
    """Flag headlines that describe an already observed stock-price move."""
    return bool(REACTION_REPORTING_PATTERN.search(normalize_text(text)))

