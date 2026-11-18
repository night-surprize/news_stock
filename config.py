"""Configuration for the Russian news-to-stock reaction project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

START_DATE = "2021-01-01"
END_DATE = "2026-10-31"
BENCHMARK = "IMOEX"
HORIZONS = (1, 3, 5)
MARKET_CLOSE_HOUR = 18
MARKET_CLOSE_MINUTE = 45
RANDOM_STATE = 42

# Only Russian issuers with shares traded on Moscow Exchange are included.
COMPANIES = {
    "PHOR": {
        "name": "ФосАгро",
        "sector": "agriculture",
        "query": "ФосАгро",
        "aliases": ["фосагро", "phosagro", "phor"],
        "business": "Фосфорные удобрения; спрос агросектора, цены на удобрения и экспортная логистика.",
        "geography": "Вологодская, Мурманская, Саратовская и Ленинградская области.",
    },
    "AKRN": {
        "name": "Акрон",
        "sector": "agriculture",
        "query": "(Акрон OR Acron) удобрения",
        "aliases": ["акрон", "acron", "akrn"],
        "business": "Минеральные удобрения; чувствительность к газу, экспортным ценам и аграрному спросу.",
        "geography": "Новгородская и Смоленская области.",
    },
    "GCHE": {
        "name": "Черкизово",
        "sector": "agriculture",
        "query": "Черкизово",
        "aliases": ["черкизово", "cherkizovo", "gche"],
        "business": "Мясная продукция и птицеводство; корма, продовольственная инфляция и потребительский спрос.",
        "geography": "Центральная Россия и несколько аграрных регионов.",
    },
    "AQUA": {
        "name": "ИНАРКТИКА",
        "sector": "agriculture",
        "query": "(Инарктика OR \"Русская аквакультура\")",
        "aliases": ["инаркатика", "инарктика", "русская аквакультура", "aqua"],
        "business": "Аквакультура; биологические риски, цены на рыбу, импорт кормов и малька.",
        "geography": "Мурманская область и Карелия.",
    },
    "LKOH": {
        "name": "ЛУКОЙЛ",
        "sector": "fuel_energy",
        "query": "Лукойл",
        "aliases": ["лукойл", "lukoil", "lkoh"],
        "business": "Нефтедобыча и переработка; нефть, налоги, санкции и экспорт.",
        "geography": "Россия и международные проекты.",
    },
    "ROSN": {
        "name": "Роснефть",
        "sector": "fuel_energy",
        "query": "Роснефть",
        "aliases": ["роснефть", "rosneft", "rosn"],
        "business": "Нефтедобыча и переработка; нефть, бюджетные налоги, санкции и капитальные проекты.",
        "geography": "Ключевые нефтедобывающие регионы России.",
    },
    "TATN": {
        "name": "Татнефть",
        "sector": "fuel_energy",
        "query": "Татнефть",
        "aliases": ["татнефть", "tatneft", "tatn"],
        "business": "Нефть и переработка; высокая региональная зависимость от Татарстана.",
        "geography": "Республика Татарстан.",
    },
    "NVTK": {
        "name": "НОВАТЭК",
        "sector": "fuel_energy",
        "query": "Новатэк",
        "aliases": ["новатэк", "novatek", "nvtk"],
        "business": "Газ и СПГ; экспортная инфраструктура, санкции, цены на газ и запуск проектов.",
        "geography": "ЯНАО, Арктика и экспортные СПГ-проекты.",
    },
    "FLOT": {
        "name": "Совкомфлот",
        "sector": "logistics",
        "query": "Совкомфлот",
        "aliases": ["совкомфлот", "sovcomflot", "flot"],
        "business": "Морские перевозки энергоносителей; фрахт, санкции и экспортные потоки.",
        "geography": "Российские порты и международные маршруты.",
    },
    "NMTP": {
        "name": "НМТП",
        "sector": "logistics",
        "query": "(НМТП OR \"Новороссийский морской торговый порт\")",
        "aliases": ["нмтп", "новороссийский морской торговый порт", "nmtp"],
        "business": "Портовая перевалка; экспорт нефти, зерна и грузооборот южных портов.",
        "geography": "Краснодарский край и Балтика.",
    },
    "FESH": {
        "name": "ДВМП (FESCO)",
        "sector": "logistics",
        "query": "(FESCO OR ДВМП)",
        "aliases": ["fesco", "двмп", "дальневосточное морское пароходство", "fesh"],
        "business": "Контейнерная логистика; торговые потоки с Азией, порты и железная дорога.",
        "geography": "Дальний Восток и трансконтинентальные маршруты.",
    },
    "AFLT": {
        "name": "Аэрофлот",
        "sector": "logistics",
        "query": "Аэрофлот",
        "aliases": ["аэрофлот", "aeroflot", "aflt"],
        "business": "Авиаперевозки; пассажиропоток, топливо, валютный курс и ограничения полетов.",
        "geography": "Федеральная маршрутная сеть и международные направления.",
    },
    "SBER": {
        "name": "Сбербанк",
        "sector": "banks",
        "query": "(Сбербанк OR Сбер)",
        "aliases": ["сбербанк", "сбер", "sber"],
        "business": "Универсальный банк; ставка ЦБ, кредитный риск, маржа, комиссии и регулирование.",
        "geography": "Вся Россия.",
    },
    "VTBR": {
        "name": "ВТБ",
        "sector": "banks",
        "query": "банк ВТБ",
        "aliases": ["втб", "vtb", "vtbr"],
        "business": "Универсальный банк; ставка ЦБ, капитал, кредитный риск и государственные решения.",
        "geography": "Вся Россия.",
    },
    "BSPB": {
        "name": "Банк Санкт-Петербург",
        "sector": "banks",
        "query": "Банк Санкт-Петербург",
        "aliases": ["банк санкт-петербург", "банк санкт петербург", "бспб", "bspb"],
        "business": "Регионально концентрированный банк; ставка, качество кредитов и экономика Северо-Запада.",
        "geography": "Санкт-Петербург, Ленинградская область и Северо-Запад.",
    },
    "CBOM": {
        "name": "Московский кредитный банк",
        "sector": "banks",
        "query": "(\"Московский кредитный банк\" OR МКБ)",
        "aliases": ["московский кредитный банк", "мкб", "credit bank of moscow", "cbom"],
        "business": "Корпоративный и розничный банк; ставка, фондирование, кредитный риск и регулирование.",
        "geography": "Москва и крупнейшие российские города.",
    },
}

NEWS_SOURCES = {
    "ria": {
        "label": "РИА Новости",
        "query": "site:ria.ru",
        "group": "federal",
    },
    "rbc": {
        "label": "РБК",
        "query": "site:rbc.ru",
        "group": "federal",
    },
    "kommersant": {
        "label": "Коммерсантъ",
        "query": "site:kommersant.ru",
        "group": "federal",
    },
    "forbes": {
        "label": "Forbes Россия",
        "query": "site:forbes.ru",
        "group": "federal",
    },
    "regional": {
        "label": "Региональные СМИ",
        "query": "(site:business-gazeta.ru OR site:interfax-russia.ru OR site:bashinform.ru OR site:ufa.rbc.ru OR site:rt.rbc.ru OR site:spb.rbc.ru)",
        "group": "regional",
    },
}

SECTOR_KEYWORDS = {
    "agriculture": ["урожай", "удобрени", "зерно", "агро", "мясо", "птиц", "рыб", "аквакультур", "продовольств"],
    "fuel_energy": ["нефть", "газ", "спг", "топлив", "добыч", "опек", "баррел", "энергоносител"],
    "logistics": ["логист", "порт", "перевоз", "фрахт", "контейнер", "авиа", "груз", "транспорт"],
    "banks": ["банк", "ставк", "кредит", "ипотек", "вклад", "ликвидност", "цб", "инфляц"],
}

