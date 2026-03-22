"""
Prop Firm Safety Module
- Drawdown tracker (daily + total)
- News blackout filter (ForexFactory)
- Position size calculator
- Trade journal
- Weekend filter
- Profit target tracker
- Risk dashboard
"""

import asyncio
import aiohttp
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Optional
from bs4 import BeautifulSoup

# ── Persistent Storage (JSON file) ───────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "prop_data.json")

def load_data() -> dict:
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "account": {
            "balance":        10000.0,
            "starting":       10000.0,
            "daily_start":    10000.0,
            "daily_date":     "",
            "peak":           10000.0,
            "profit_target":  1000.0,   # 10% default
            "max_daily_dd":   500.0,    # 5% default
            "max_total_dd":   1000.0,   # 10% default
            "risk_per_trade": 1.0,      # 1% default
            "firm":           "Generic",
            "phase":          "Challenge",
        },
        "trades": [],
        "daily_pnl": 0.0,
        "total_pnl": 0.0,
        "trading_days": [],
        "status": "ACTIVE",  # ACTIVE, PAUSED, BLOWN, PASSED
    }

def save_data(data: dict):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_data() -> dict:
    data = load_data()
    # Reset daily PnL if new day
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if data["account"]["daily_date"] != today:
        data["account"]["daily_date"]  = today
        data["account"]["daily_start"] = data["account"]["balance"]
        data["daily_pnl"]              = 0.0
        if today not in data["trading_days"]:
            data["trading_days"].append(today)
        save_data(data)
    return data


# ── News Blackout Engine ──────────────────────────────────────────────────────

HIGH_IMPACT_KEYWORDS = [
    "non-farm", "nfp", "fomc", "fed", "interest rate", "cpi", "gdp",
    "unemployment", "payroll", "inflation", "boe", "ecb", "rba", "boj",
    "rbnz", "snb", "bank of canada", "pmi", "retail sales", "trade balance",
]

CURRENCY_MAP = {
    "USD": ["EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD",
            "XAUUSD","XAGUSD"],
    "EUR": ["EURUSD","EURGBP","EURJPY","EURCHF","EURAUD","EURCAD"],
    "GBP": ["GBPUSD","EURGBP","GBPJPY","GBPAUD","GBPCAD","GBPCHF"],
    "JPY": ["USDJPY","EURJPY","GBPJPY","AUDJPY","CADJPY","CHFJPY","NZDJPY"],
    "AUD": ["AUDUSD","EURAUD","GBPAUD","AUDJPY","AUDCAD","AUDCHF","AUDNZD"],
    "CAD": ["USDCAD","EURCAD","GBPCAD","CADJPY","AUDCAD"],
    "CHF": ["USDCHF","EURCHF","GBPCHF","CHFJPY","AUDCHF"],
    "NZD": ["NZDUSD","NZDJPY","AUDNZD"],
}


async def get_news_blackout(pair: str, buffer_minutes: int = 5) -> dict:
    """
    Check if current time is within news blackout window for a pair.
    Returns: {blocked: bool, reason: str, next_clear: str}
    """
    now = datetime.now(timezone.utc)

    # Weekend filter
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return {
            "blocked": True,
            "reason":  "Weekend — market closed",
            "next_clear": "Sunday 21:00 UTC",
        }
    # Friday close
    if now.weekday() == 4 and now.hour >= 21:
        return {
            "blocked": True,
            "reason":  "Friday close — avoid weekend gap risk",
            "next_clear": "Monday 00:00 UTC",
        }

    # Try to get news from ForexFactory
    try:
        events = await _fetch_forexfactory_events()
        pair_currencies = _get_pair_currencies(pair)

        for event in events:
            if event.get("currency") not in pair_currencies:
                continue
            if event.get("impact") != "HIGH":
                continue

            event_time = event.get("datetime")
            if not event_time:
                continue

            diff = abs((event_time - now).total_seconds() / 60)
            if diff <= buffer_minutes:
                return {
                    "blocked": True,
                    "reason":  f"🔴 High impact news: {event['title']} ({event['currency']}) in {int(diff)} min",
                    "next_clear": (event_time + timedelta(minutes=buffer_minutes)).strftime("%H:%M UTC"),
                }

    except Exception:
        pass

    return {"blocked": False, "reason": "Clear", "next_clear": "Now"}


async def _fetch_forexfactory_events() -> list:
    """Scrape ForexFactory for today's high-impact events with times."""
    url = "https://www.forexfactory.com/calendar"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    events  = []
    today   = datetime.now(timezone.utc).date()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers,
                                   timeout=aiohttp.ClientTimeout(total=10)) as resp:
                html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")
        rows = soup.select("tr.calendar__row")

        for row in rows:
            try:
                impact = row.select_one(".calendar__impact span")
                if not impact or "high" not in str(impact.get("class", [])).lower():
                    continue

                currency_el = row.select_one(".calendar__currency")
                title_el    = row.select_one(".calendar__event-title")
                time_el     = row.select_one(".calendar__time")

                if not currency_el or not title_el:
                    continue

                time_str = time_el.text.strip() if time_el else ""
                event_dt = None

                if time_str and ":" in time_str:
                    try:
                        t = datetime.strptime(time_str, "%I:%M%p").replace(
                            year=today.year, month=today.month, day=today.day,
                            tzinfo=timezone.utc
                        )
                        event_dt = t
                    except Exception:
                        pass

                events.append({
                    "currency": currency_el.text.strip(),
                    "title":    title_el.text.strip(),
                    "impact":   "HIGH",
                    "datetime": event_dt,
                })
            except Exception:
                continue
    except Exception:
        pass

    return events


def _get_pair_currencies(pair: str) -> list:
    base  = pair[:3]
    quote = pair[3:]
    return [base, quote]


# ── Position Size Calculator ──────────────────────────────────────────────────

PIP_SIZE = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDCHF": 0.0001,
    "AUDUSD": 0.0001, "USDCAD": 0.0001, "NZDUSD": 0.0001,
    "EURGBP": 0.0001, "EURCHF": 0.0001, "EURAUD": 0.0001,
    "EURCAD": 0.0001, "GBPAUD": 0.0001, "GBPCAD": 0.0001,
    "GBPCHF": 0.0001, "AUDCAD": 0.0001, "AUDCHF": 0.0001,
    "AUDNZD": 0.0001,
    "USDJPY": 0.01,   "EURJPY": 0.01,   "GBPJPY": 0.01,
    "AUDJPY": 0.01,   "CADJPY": 0.01,   "CHFJPY": 0.01, "NZDJPY": 0.01,
    "XAUUSD": 0.10,   "XAGUSD": 0.01,
}

PIP_VALUE_USD = {
    # Approximate pip value per standard lot in USD
    "EURUSD": 10.0, "GBPUSD": 10.0, "AUDUSD": 10.0, "NZDUSD": 10.0,
    "USDJPY": 9.1,  "USDCHF": 10.0, "USDCAD": 7.5,
    "EURGBP": 12.5, "EURJPY": 9.1,  "GBPJPY": 9.1,  "AUDJPY": 9.1,
    "CADJPY": 9.1,  "CHFJPY": 9.1,  "NZDJPY": 9.1,
    "EURCHF": 10.0, "EURAUD": 10.0, "EURCAD": 7.5,
    "GBPAUD": 10.0, "GBPCAD": 7.5,  "GBPCHF": 10.0,
    "AUDCAD": 7.5,  "AUDCHF": 10.0, "AUDNZD": 10.0,
    "XAUUSD": 10.0, "XAGUSD": 50.0,
}


def calculate_lot_size(pair: str, entry: float, sl: float,
                        account_balance: float, risk_pct: float) -> dict:
    """
    Calculate proper lot size based on account risk %.
    Returns lot size, risk amount, and pip count.
    """
    pip   = PIP_SIZE.get(pair, 0.0001)
    pv    = PIP_VALUE_USD.get(pair, 10.0)

    sl_pips      = abs(entry - sl) / pip
    risk_amount  = account_balance * (risk_pct / 100)
    lot_size     = risk_amount / (sl_pips * pv)
    lot_size     = max(0.01, round(lot_size, 2))  # min 0.01 lots

    # Cap at reasonable max
    lot_size = min(lot_size, 10.0)

    return {
        "lot_size":    lot_size,
        "risk_amount": round(risk_amount, 2),
        "risk_pct":    risk_pct,
        "sl_pips":     round(sl_pips, 1),
        "pip_value":   pv,
    }


# ── Drawdown Tracker ──────────────────────────────────────────────────────────

class DrawdownTracker:
    def __init__(self):
        pass

    def check_limits(self, data: dict) -> dict:
        """
        Check if trading should be allowed based on drawdown limits.
        Returns: {allowed: bool, reason: str, daily_dd: float, total_dd: float}
        """
        acc         = data["account"]
        balance     = acc["balance"]
        daily_start = acc["daily_start"]
        starting    = acc["starting"]
        peak        = acc["peak"]

        daily_dd  = daily_start - balance
        total_dd  = peak - balance
        daily_pct = (daily_dd / daily_start * 100) if daily_start > 0 else 0
        total_pct = (total_dd / peak * 100) if peak > 0 else 0

        max_daily = acc["max_daily_dd"]
        max_total = acc["max_total_dd"]

        if total_dd >= max_total:
            return {
                "allowed":   False,
                "reason":    f"❌ TOTAL DRAWDOWN LIMIT HIT ({total_pct:.1f}%) — Account protection active",
                "daily_dd":  round(daily_dd, 2),
                "total_dd":  round(total_dd, 2),
                "daily_pct": round(daily_pct, 1),
                "total_pct": round(total_pct, 1),
            }

        if daily_dd >= max_daily:
            return {
                "allowed":   False,
                "reason":    f"⛔ DAILY DRAWDOWN LIMIT HIT ({daily_pct:.1f}%) — Wait for next day",
                "daily_dd":  round(daily_dd, 2),
                "total_dd":  round(total_dd, 2),
                "daily_pct": round(daily_pct, 1),
                "total_pct": round(total_pct, 1),
            }

        # Warning zones
        daily_warn = daily_dd >= max_daily * 0.7
        total_warn = total_dd >= max_total * 0.7

        reason = "✅ Within limits"
        if total_warn:
            reason = f"⚠️ Approaching total DD limit ({total_pct:.1f}%)"
        elif daily_warn:
            reason = f"⚠️ Approaching daily DD limit ({daily_pct:.1f}%)"

        return {
            "allowed":   True,
            "reason":    reason,
            "daily_dd":  round(daily_dd, 2),
            "total_dd":  round(total_dd, 2),
            "daily_pct": round(daily_pct, 1),
            "total_pct": round(total_pct, 1),
        }

    def update_balance(self, pnl: float):
        """Update balance after a trade closes."""
        data             = get_data()
        data["account"]["balance"] += pnl
        data["daily_pnl"]          += pnl
        data["total_pnl"]          += pnl

        # Update peak
        if data["account"]["balance"] > data["account"]["peak"]:
            data["account"]["peak"] = data["account"]["balance"]

        # Log trade
        data["trades"].append({
            "time":    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "pnl":     round(pnl, 2),
            "balance": round(data["account"]["balance"], 2),
        })

        # Check if blown
        total_dd = data["account"]["peak"] - data["account"]["balance"]
        if total_dd >= data["account"]["max_total_dd"]:
            data["status"] = "BLOWN"

        # Check if target hit
        profit = data["account"]["balance"] - data["account"]["starting"]
        if profit >= data["account"]["profit_target"]:
            data["status"] = "PASSED"

        save_data(data)
        return data

    def get_dashboard(self) -> str:
        """Generate prop firm dashboard text."""
        data   = get_data()
        acc    = data["account"]
        limits = self.check_limits(data)

        balance      = acc["balance"]
        starting     = acc["starting"]
        peak         = acc["peak"]
        profit       = balance - starting
        profit_pct   = profit / starting * 100
        target       = acc["profit_target"]
        target_pct   = acc["profit_target"] / starting * 100
        progress_pct = (profit / target * 100) if target > 0 else 0
        progress_bar = "█" * int(min(progress_pct, 100) / 10) + "░" * (10 - int(min(progress_pct, 100) / 10))

        daily_dd   = limits["daily_dd"]
        total_dd   = limits["total_dd"]
        daily_pct  = limits["daily_pct"]
        total_pct  = limits["total_pct"]
        max_daily  = acc["max_daily_dd"]
        max_total  = acc["max_total_dd"]

        daily_bar  = "█" * int(min(daily_pct / (max_daily/starting*100) * 10, 10))
        daily_bar += "░" * (10 - len(daily_bar))
        total_bar  = "█" * int(min(total_pct / (max_total/starting*100) * 10, 10))
        total_bar += "░" * (10 - len(total_bar))

        status_emoji = {
            "ACTIVE": "🟢", "PAUSED": "🟡",
            "BLOWN":  "🔴", "PASSED": "🏆"
        }.get(data["status"], "⚪")

        trades_today = [t for t in data["trades"]
                        if t["time"].startswith(acc["daily_date"])]

        return (
            f"📊 *Prop Firm Dashboard*\n"
            f"🏢 Firm: {acc['firm']} | Phase: {acc['phase']}\n"
            f"{status_emoji} Status: {data['status']}\n\n"
            f"💰 *Account*\n"
            f"Balance:  `${balance:,.2f}`\n"
            f"Profit:   `${profit:+,.2f}` ({profit_pct:+.2f}%)\n"
            f"Peak:     `${peak:,.2f}`\n\n"
            f"🎯 *Profit Target*\n"
            f"Target: `${target:,.2f}` ({target_pct:.1f}%)\n"
            f"Progress: `{progress_bar}` {progress_pct:.1f}%\n\n"
            f"🛡️ *Drawdown Limits*\n"
            f"Daily DD:  `{daily_bar}` ${daily_dd:.2f} / ${max_daily:.2f} ({daily_pct:.1f}%)\n"
            f"Total DD:  `{total_bar}` ${total_dd:.2f} / ${max_total:.2f} ({total_pct:.1f}%)\n\n"
            f"📈 *Activity*\n"
            f"Trading Days: `{len(data['trading_days'])}`\n"
            f"Trades Today: `{len(trades_today)}`\n"
            f"Total Trades: `{len(data['trades'])}`\n\n"
            f"⚡ *Risk per Trade*: `{acc['risk_per_trade']}%`\n"
            f"🔒 *Limit Status*: {limits['reason']}"
        )


# ── Trade Journal ─────────────────────────────────────────────────────────────

def log_signal(pair: str, direction: str, entry: float, sl: float,
               tp1: float, tp2: float, confidence: float, lot_size: float):
    """Log a signal to the trade journal."""
    data = get_data()
    data["trades"].append({
        "time":       datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "pair":       pair,
        "direction":  direction,
        "entry":      entry,
        "sl":         sl,
        "tp1":        tp1,
        "tp2":        tp2,
        "confidence": confidence,
        "lot_size":   lot_size,
        "pnl":        None,  # filled when trade closes
        "status":     "OPEN",
    })
    save_data(data)


def get_journal_text(last_n: int = 10) -> str:
    """Get recent trade journal entries."""
    data   = get_data()
    trades = [t for t in data["trades"] if "pair" in t]
    trades = trades[-last_n:]

    if not trades:
        return "📓 *Trade Journal*\n\nNo trades logged yet."

    lines = ["📓 *Trade Journal* — Last 10 Signals\n"]
    wins = losses = 0
    for t in reversed(trades):
        status = t.get("status", "OPEN")
        pnl    = t.get("pnl")
        pnl_str = f"`${pnl:+.2f}`" if pnl is not None else "`OPEN`"
        emoji  = "🟢" if status == "WIN" else "🔴" if status == "LOSS" else "🟡"
        lines.append(
            f"{emoji} {t.get('pair','')} {t.get('direction','')} "
            f"@ `{t.get('entry','')}` | {pnl_str} | {t.get('time','')[:10]}"
        )
        if status == "WIN":   wins += 1
        if status == "LOSS":  losses += 1

    total = wins + losses
    wr    = wins / total * 100 if total > 0 else 0
    lines.append(f"\n📊 Win Rate: `{wr:.1f}%` ({wins}W / {losses}L)")
    return "\n".join(lines)


# ── Global instances ──────────────────────────────────────────────────────────
dd_tracker = DrawdownTracker()
