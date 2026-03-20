"""
News Engine — Forex News & Sentiment
Scrapes ForexFactory economic calendar headlines and
aggregates sentiment for each currency pair.
"""

import aiohttp
import asyncio
from datetime import datetime, date
from bs4 import BeautifulSoup
from typing import List, Dict


PAIR_CURRENCIES = {
    "EURUSD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"],
    "XAUUSD": ["XAU", "USD"],
}

BULLISH_WORDS = [
    "beat", "better", "surge", "rise", "rally", "strong", "growth",
    "higher", "positive", "optimistic", "hawkish", "rate hike", "improve",
]
BEARISH_WORDS = [
    "miss", "worse", "fall", "drop", "decline", "weak", "contraction",
    "lower", "negative", "pessimistic", "dovish", "rate cut", "disappoint",
]


class NewsEngine:
    def __init__(self):
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

    async def fetch_forexfactory_news(self) -> List[Dict]:
        """Scrape ForexFactory calendar for today's high-impact events."""
        url = "https://www.forexfactory.com/calendar"
        try:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    html = await resp.text()

            soup = BeautifulSoup(html, "html.parser")
            events = []
            rows = soup.select("tr.calendar__row")

            for row in rows:
                impact = row.select_one(".calendar__impact span")
                if not impact:
                    continue
                impact_class = impact.get("class", [])
                # Only high/medium impact
                if not any(c in str(impact_class) for c in ["high", "medium"]):
                    continue

                currency_el = row.select_one(".calendar__currency")
                title_el = row.select_one(".calendar__event-title")
                actual_el = row.select_one(".calendar__actual")
                forecast_el = row.select_one(".calendar__forecast")

                if not currency_el or not title_el:
                    continue

                events.append({
                    "currency": currency_el.text.strip(),
                    "title": title_el.text.strip(),
                    "actual": actual_el.text.strip() if actual_el else "",
                    "forecast": forecast_el.text.strip() if forecast_el else "",
                })

            return events

        except Exception:
            return self._fallback_news()

    def _fallback_news(self) -> List[Dict]:
        """Return placeholder events when scraping fails."""
        return [
            {"currency": "USD", "title": "Fed Speakers Today", "actual": "", "forecast": ""},
            {"currency": "EUR", "title": "ECB Policy Meeting Minutes", "actual": "", "forecast": ""},
            {"currency": "GBP", "title": "UK CPI Data", "actual": "", "forecast": ""},
            {"currency": "JPY", "title": "BoJ Policy Statement", "actual": "", "forecast": ""},
        ]

    def score_sentiment(self, events: List[Dict], currency: str) -> int:
        """Returns sentiment score for a currency: positive=bullish, negative=bearish."""
        score = 0
        for event in events:
            if event["currency"].upper() != currency.upper():
                continue
            text = (event["title"] + " " + event["actual"]).lower()
            for word in BULLISH_WORDS:
                if word in text:
                    score += 1
            for word in BEARISH_WORDS:
                if word in text:
                    score -= 1
            # Actual vs forecast comparison
            try:
                actual = float(event["actual"].replace("%", "").replace("K", "").replace("M", ""))
                forecast = float(event["forecast"].replace("%", "").replace("K", "").replace("M", ""))
                if actual > forecast:
                    score += 1
                elif actual < forecast:
                    score -= 1
            except (ValueError, AttributeError):
                pass
        return score

    def pair_sentiment(self, events: List[Dict], pair: str) -> str:
        currencies = PAIR_CURRENCIES.get(pair, [])
        if len(currencies) < 2:
            return "Neutral 🟡"

        base_score = self.score_sentiment(events, currencies[0])
        quote_score = self.score_sentiment(events, currencies[1])
        net = base_score - quote_score

        if net >= 2:
            return "Bullish 🟢"
        elif net <= -2:
            return "Bearish 🔴"
        else:
            return "Neutral 🟡"

    async def get_forex_news(self) -> str:
        events = await self.fetch_forexfactory_news()
        if not events:
            return "📰 No high-impact events found for today."

        today = date.today().strftime("%A, %B %d %Y")
        lines = [f"📰 *Forex Calendar — {today}*\n"]

        currency_events: Dict[str, List] = {}
        for e in events:
            c = e["currency"]
            if c not in currency_events:
                currency_events[c] = []
            currency_events[c].append(e)

        for currency, evs in sorted(currency_events.items()):
            lines.append(f"*{currency}*")
            for ev in evs:
                actual_str = f" | Actual: `{ev['actual']}`" if ev["actual"] else ""
                forecast_str = f" | Forecast: `{ev['forecast']}`" if ev["forecast"] else ""
                lines.append(f"  • {ev['title']}{actual_str}{forecast_str}")
            lines.append("")

        lines.append("─────────────────")
        lines.append("*Pair Sentiment:*")
        for pair in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]:
            sentiment = self.pair_sentiment(events, pair)
            display = pair[:3] + "/" + pair[3:]
            lines.append(f"  {display}: {sentiment}")

        return "\n".join(lines)
