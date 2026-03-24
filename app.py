"""
Forex Signal Bot v2 — app.py
Features:
- On-demand signals for all 25 pairs
- Best picks notification (top 3 signals every N hours)
- 24/7 via GitHub Actions cron
"""

import asyncio
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from signals import SignalEngine, ScalpingEngine, ArbitrageEngine
from propfirm import (dd_tracker, get_data, save_data, get_news_blackout,
                       calculate_lot_size, log_signal, get_journal_text)
from news import NewsEngine
from config import Config

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

config        = Config()
signal_engine = SignalEngine(config.TWELVEDATA_API_KEY)
scalp_engine  = ScalpingEngine(config.TWELVEDATA_API_KEY)
arb_engine    = ArbitrageEngine(config.TWELVEDATA_API_KEY)
news_engine   = NewsEngine()

# Cached news events (refreshed every 30 min)
_news_events_cache = []
_news_events_ts    = 0

MAJORS      = ["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD"]
MINORS      = ["EURGBP","EURJPY","GBPJPY","AUDJPY","CADJPY","CHFJPY",
               "EURCHF","EURAUD","EURCAD","GBPAUD","GBPCAD","GBPCHF",
               "AUDCAD","AUDCHF","AUDNZD","NZDJPY"]
COMMODITIES = ["XAUUSD","XAGUSD"]
PAIRS       = MAJORS + MINORS + COMMODITIES


# ── Formatters ────────────────────────────────────────────────────────────────

def format_signal(s: dict) -> str:
    if s["direction"] == "N/A":
        reason = s.get("news_sentiment","").replace("❌ ","")
        return f"⚠️ *{s['pair']}* — {reason}"
    dir_emoji = "🟢 BUY" if s["direction"] == "BUY" else "🔴 SELL"
    conf      = s["confidence"]
    bar       = "█" * int(conf / 10) + "░" * (10 - int(conf / 10))
    sess_icon = "✅" if s.get("in_session") else "⏸"
    ind       = s.get("indicators", {})
    ml_tag    = " _(ML)_" if s.get("ml_used") else ""
    return (
        f"*{s['pair']}* — {dir_emoji}\n"
        f"🕐 `{s['timestamp']}`\n\n"
        f"💰 Entry: `{s['entry']}`\n"
        f"🛑 SL:    `{s['sl']}`\n"
        f"🎯 TP1:   `{s['tp1']}`\n"
        f"🎯 TP2:   `{s['tp2']}`\n\n"
        f"📊 Confidence: {conf}%{ml_tag} `{bar}`\n"
        f"🏆 Quality: {s.get('quality','N/A')}\n"
        f"📈 H4 Trend: {s.get('h4_trend','N/A')}\n"
        f"{sess_icon} Session: {s.get('session','N/A')}\n\n"
        f"RSI: `{ind.get('rsi',0):.1f}` {ind.get('rsi_signal','')}\n"
        f"MACD: {ind.get('macd_signal','')}\n"
        f"EMA: {ind.get('ema_cross','')}\n"
        f"Stoch: {ind.get('stochastic','')}\n"
        f"ADX: {ind.get('adx','')}\n"
        f"ATR: `{ind.get('atr',0)}`\n\n"
        f"📰 News: {s.get('news_sentiment','N/A')}\n"
        f"⚠️ _Use proper risk management._"
    )


def format_bestpick(s: dict, rank: int) -> str:
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    medal  = medals.get(rank, f"#{rank}")
    dir_emoji = "🟢 BUY" if s["direction"] == "BUY" else "🔴 SELL"
    conf   = s["confidence"]
    bar    = "█" * int(conf / 10) + "░" * (10 - int(conf / 10))
    return (
        f"{medal} *{s['pair']}* — {dir_emoji}\n"
        f"💰 Entry: `{s['entry']}` | SL: `{s['sl']}`\n"
        f"🎯 TP1: `{s['tp1']}` | TP2: `{s['tp2']}`\n"
        f"📊 Confidence: {conf}% `{bar}`\n"
        f"🏆 {s.get('quality','N/A')} | H4: {s.get('h4_trend','N/A')}\n"
        f"⚠️ _Manage your risk._"
    )


def format_scalp(s: dict) -> str:
    if s["direction"] == "N/A":
        err = s.get("indicators", {}).get("error", "Error")
        pair = s["pair"]
        return f"⚠️ *{pair}* SCALP — {err}"
    dir_emoji = "🟢 BUY" if s["direction"] == "BUY" else "🔴 SELL"
    conf  = s["confidence"]
    bar   = "█" * int(conf/10) + "░" * (10 - int(conf/10))
    ind   = s.get("indicators", {})
    pair  = s["pair"]
    ts    = s["timestamp"]
    tf    = s["timeframe"]
    entry = s["entry"]
    sl    = s["sl"]
    tp1   = s["tp1"]
    tp2   = s["tp2"]
    qual  = s.get("quality", "N/A")
    vwap  = ind.get("vwap", "")
    mom   = ind.get("momentum", "")
    flow  = ind.get("tick_flow", "")
    rib   = ind.get("ribbon", "")
    eng   = ind.get("engulfing", "")
    pin   = ind.get("pin_bar", "")
    m15   = ind.get("m15_trend", "")
    sprd  = ind.get("spread", "")
    sup   = ind.get("micro_sup", "")
    res   = ind.get("micro_res", "")
    lines = [
        f"⚡ *{pair}* SCALP — {dir_emoji}",
        f"🕐 `{ts}`",
        f"📋 `{tf}`",
        "",
        f"💰 Entry: `{entry}`",
        f"🛑 SL:    `{sl}`",
        f"🎯 TP1:   `{tp1}`",
        f"🎯 TP2:   `{tp2}`",
        "",
        f"📊 Confidence: {conf}% `{bar}`",
        f"🏆 {qual}",
        "",
        f"VWAP: {vwap}",
        f"Momentum: {mom}",
        f"Tick Flow: {flow}",
        f"EMA Ribbon: {rib}",
        f"Engulfing: {eng}",
        f"Pin Bar: {pin}",
        f"M15 Trend: {m15}",
        f"Spread: {sprd}",
        f"Support: `{sup}` | Resistance: `{res}`",
        "",
        "⚠️ _Scalp: tight SL, quick TP. Max 15 min hold._",
    ]
    return "\n".join(lines)


def format_arb(a: dict) -> str:
    name_a  = a["name_a"]
    name_b  = a["name_b"]
    act_a   = a["action_a"]
    act_b   = a["action_b"]
    ts      = a["timestamp"]
    corr    = a["correlation"]
    zsc     = a["zscore"]
    desc    = a["description"]
    conf    = a["confidence"]
    lines = [
        "🔄 *ARBITRAGE OPPORTUNITY*",
        f"🕐 `{ts}`",
        "",
        f"Pair A: *{name_a}* → `{act_a}`",
        f"Pair B: *{name_b}* → `{act_b}`",
        "",
        f"📊 Correlation: `{corr}`",
        f"📈 Z-Score: `{zsc}` (signal at ±2.0)",
        f"💡 {desc}",
        f"📊 Confidence: `{conf}%`",
        "",
        "⚠️ _Stat arb: trade both pairs simultaneously. Equal position sizes._",
    ]
    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def safe_send(send_fn, text: str, **kwargs):
    if len(text) <= 4096:
        await send_fn(text, **kwargs)
    else:
        for i in range(0, len(text), 4096):
            await send_fn(text[i:i+4096], **kwargs)


async def send_signals(send_fn, pairs: list):
    """Send signals one by one. 8s delay inside get_signal respects TD rate limits."""
    for i, pair in enumerate(pairs):
        if i > 0:
            await send_fn(f"⏳ Fetching {pair}... ({i+1}/{len(pairs)})")
        result = await signal_engine.get_signal(pair)
        await safe_send(send_fn, format_signal(result), parse_mode="Markdown")


async def get_best_picks(pairs: list, top_n: int = 3) -> list:
    """Scan all pairs and return top N by confidence, HIGH/MEDIUM quality only."""
    results = []
    for pair in pairs:
        result = await signal_engine.get_signal(pair)
        if result["direction"] != "N/A" and "LOW" not in result.get("quality", ""):
            results.append(result)
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results[:top_n]


# ── Prop Firm Safety Check ────────────────────────────────────────────────────

async def get_news_events() -> list:
    """Get cached news events, refresh if older than 30 min."""
    global _news_events_cache, _news_events_ts
    import time as _t
    if _t.time() - _news_events_ts > 1800:
        events = await NewsFilter.get_high_impact_events()
        if events:
            _news_events_cache = events
            _news_events_ts    = _t.time()
    return _news_events_cache


async def safety_check(chat_id: int, pair: str) -> tuple:
    account = load_account(chat_id)
    account = DrawdownTracker.reset_daily_if_needed(account)
    account = DrawdownTracker.check_limits(account)
    save_account(chat_id, account)

    if account["status"] == "DAILY_LIMIT":
        return False, (
            "🛑 *Daily Drawdown Limit Hit*\n\n"
            f"You have reached your {account['max_daily_dd_pct']}% daily loss limit.\n"
            "No more signals today. Reset tomorrow."
        )
    if account["status"] == "TOTAL_LIMIT":
        return False, (
            "❌ *Total Drawdown Limit Hit*\n\n"
            f"You have reached the {account['max_total_dd_pct']}% total drawdown limit.\n"
            "Account is at risk. Stop trading immediately."
        )
    if account["status"] == "TARGET_HIT":
        return False, (
            "🎯 *Profit Target Reached!*\n\n"
            f"You have hit your {account['profit_target_pct']}% profit target!\n"
            "Consider stopping or switching to conservative mode."
        )

    is_weekend, wk_reason = NewsFilter.is_weekend()
    if is_weekend:
        return False, f"*Weekend Filter*\n\n{wk_reason}"

    if pair and pair != "ALL":
        events = await get_news_events()
        is_blocked, news_reason = NewsFilter.is_blackout(events, pair)
        if is_blocked:
            return False, f"*News Blackout Active*\n\n{news_reason}\n\nWait 5 minutes."

    return True, ""


def format_signal_with_sizing(s: dict, chat_id: int) -> str:
    account = load_account(chat_id)
    if s["direction"] == "N/A" or s["entry"] == "N/A":
        return format_signal(s)

    try:
        from signals import PIP_SIZE
        sizing = PositionSizer.calculate(account, s["entry"], s["sl"],
                                          list(PIP_SIZE.keys())[list(PIP_SIZE.values()).index(
                                              PIP_SIZE.get(s["pair"].replace("/",""), 0.0001)
                                          )])
        sizing_line = (
            f"\n💼 *Position Size:* `{sizing['lots']} lots`\n"
            f"💸 Risk: `${sizing['risk_usd']}` ({sizing['risk_pct']}%)"
        )
    except Exception:
        sizing_line = ""

    pair_key = s["pair"].replace("/", "")
    try:
        events   = _news_events_cache
        upcoming = NewsFilter.upcoming_events(events, pair_key)
        if upcoming:
            news_warn = "\n⚠️ *Upcoming News:*\n"
            for ev in upcoming[:2]:
                news_warn += f"  • {ev['title']} ({ev['currency']}) in {ev['in_mins']} min\n"
        else:
            news_warn = ""
    except Exception:
        news_warn = ""

    base = format_signal(s)
    return base + sizing_line + news_warn


# ── Keyboards ─────────────────────────────────────────────────────────────────

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 All Signals",    callback_data="signals_all")],
        [InlineKeyboardButton("💱 Majors",          callback_data="menu_majors"),
         InlineKeyboardButton("🔀 Minors",          callback_data="menu_minors")],
        [InlineKeyboardButton("🥇 Commodities",     callback_data="menu_commodities"),
         InlineKeyboardButton("🏆 Best Picks",      callback_data="bestpicks")],
        [InlineKeyboardButton("⚡ Scalp",            callback_data="menu_scalp"),
         InlineKeyboardButton("🔄 Arbitrage",        callback_data="arb")],
        [InlineKeyboardButton("🛡️ Prop Firm",       callback_data="menu_propfirm"),
         InlineKeyboardButton("📰 News",             callback_data="news")],
        [InlineKeyboardButton("ℹ️ Help",            callback_data="help")],
    ])

def majors_keyboard():
    pairs = ["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD"]
    rows  = [[InlineKeyboardButton(p[:3]+"/"+p[3:], callback_data=f"signal_{p}")] for p in pairs]
    rows.append([InlineKeyboardButton("🔙 Back", callback_data="back_main"),
                 InlineKeyboardButton("📊 All Majors", callback_data="signals_majors")])
    return InlineKeyboardMarkup(rows)

def minors_keyboard():
    pairs = ["EURGBP","EURJPY","GBPJPY","AUDJPY","CADJPY","CHFJPY",
             "EURCHF","EURAUD","EURCAD","GBPAUD","GBPCAD","GBPCHF",
             "AUDCAD","AUDCHF","AUDNZD","NZDJPY"]
    rows = []
    for i in range(0, len(pairs), 2):
        row = []
        for p in pairs[i:i+2]:
            row.append(InlineKeyboardButton(p[:3]+"/"+p[3:], callback_data=f"signal_{p}"))
        rows.append(row)
    rows.append([InlineKeyboardButton("🔙 Back", callback_data="back_main"),
                 InlineKeyboardButton("📊 All Minors", callback_data="signals_minors")])
    return InlineKeyboardMarkup(rows)

def commodities_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🥇 XAU/USD (Gold)",   callback_data="signal_XAUUSD")],
        [InlineKeyboardButton("🥈 XAG/USD (Silver)",  callback_data="signal_XAGUSD")],
        [InlineKeyboardButton("📊 Both",              callback_data="signals_commodities")],
        [InlineKeyboardButton("🔙 Back",              callback_data="back_main")],
    ])


# ── Commands ──────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Forex Signal Bot v2*\n\n"
        "✅ H1 + H4 multi-timeframe\n"
        "✅ London / NY session filter\n"
        "✅ RSI, MACD, EMA, BB, Stoch, ADX, ATR\n"
        "✅ ML confidence scoring\n"
        "✅ 25 currency pairs\n"
        "✅ Best picks auto-notification\n\n"
        "Select an option:",
        parse_mode="Markdown",
        reply_markup=main_keyboard(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *Commands*\n\n"
        "/start — Main menu\n"
        "/signal EURUSD — Single pair\n"
        "/signals — All 25 pairs\n"
        "/majors — Major pairs\n"
        "/minors — Minor crosses\n"
        "/commodities — Gold & Silver\n"
        "/bestpicks — Top 3 signals right now\n"
        "/subscribe_bestpicks 4 — Auto best picks every 4 hrs\n"
        "/unsubscribe_bestpicks — Stop auto best picks\n"
        "/subscribe — Hourly all signals\n"
        "/unsubscribe — Stop hourly signals\n"
        "/news — Economic calendar\n\n"
        "⭐ Quality Guide:\n"
        "`⭐⭐⭐ HIGH`  → Trade full size\n"
        "`⭐⭐ MEDIUM` → Trade half size\n"
        "`⚠️ LOW`     → Skip",
        parse_mode="Markdown",
    )


async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /signal EURUSD")
        return
    pair = context.args[0].upper()
    if pair not in PAIRS:
        await update.message.reply_text(f"❌ Unknown pair: {pair}")
        return
    data   = get_data()
    limits = dd_tracker.check_limits(data)
    if not limits["allowed"]:
        await update.message.reply_text(
            f"🛡️ *Trading Halted*\n\n{limits['reason']}\n\nUse /dashboard for details.",
            parse_mode="Markdown"
        )
        return

    blackout = await get_news_blackout(pair)
    if blackout["blocked"]:
        await update.message.reply_text(
            f"📰 *News Blackout*\n\n{blackout['reason']}\n"
            f"Trading resumes: {blackout['next_clear']}",
            parse_mode="Markdown"
        )
        return

    msg    = await update.message.reply_text(f"⏳ Analyzing {pair}...")
    result = await signal_engine.get_signal(pair)

    if result["direction"] != "N/A":
        acc    = data["account"]
        ls     = calculate_lot_size(pair, result["entry"], result["sl"],
                                    acc["balance"], acc["risk_per_trade"])
        result["lot_size"]    = ls["lot_size"]
        result["risk_amount"] = ls["risk_amount"]
        result["sl_pips"]     = ls["sl_pips"]
        log_signal(pair, result["direction"], result["entry"], result["sl"],
                   result["tp1"], result["tp2"], result["confidence"], ls["lot_size"])

    await msg.edit_text(format_signal(result), parse_mode="Markdown")


async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Scanning all 25 pairs...")
    await send_signals(update.message.reply_text, PAIRS)


async def majors_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Scanning major pairs...")
    await send_signals(update.message.reply_text, MAJORS)


async def minors_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Scanning minor pairs...")
    await send_signals(update.message.reply_text, MINORS)


async def commodities_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Scanning commodities...")
    await send_signals(update.message.reply_text, COMMODITIES)


async def bestpicks_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Scanning all 25 pairs for best picks...")
    picks = await get_best_picks(PAIRS)
    if not picks:
        await msg.edit_text("😴 No HIGH/MEDIUM quality signals right now.\nMarket may be off-session or ranging. Try again later.")
        return
    await msg.edit_text(f"🏆 *Top {len(picks)} Best Picks Right Now*", parse_mode="Markdown")
    for i, pick in enumerate(picks, 1):
        await safe_send(update.message.reply_text, format_bestpick(pick, i), parse_mode="Markdown")


async def scalp_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Usage: /scalp EURUSD\nBest pairs: EURUSD, GBPUSD, USDJPY, XAUUSD"
        )
        return
    pair = context.args[0].upper()
    if pair not in PAIRS:
        await update.message.reply_text(f"❌ Unknown pair: {pair}")
        return
    msg = await update.message.reply_text(f"⚡ Analyzing {pair} M5+M15 scalp setup...")
    result = await scalp_engine.get_scalp_signal(pair)
    await msg.edit_text(format_scalp(result), parse_mode="Markdown")


async def arb_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔄 Scanning arbitrage opportunities across correlated pairs...")
    opps = await arb_engine.scan_all()
    if not opps:
        await msg.edit_text(
            "🔄 *No Arbitrage Opportunities*\n\nAll correlated pairs within normal range.\nZ-score needs ±2.0 for a signal. Try again in 30-60 min.",
            parse_mode="Markdown"
        )
        return
    await msg.edit_text(f"🔄 *{len(opps)} Arbitrage Signal(s) Found*", parse_mode="Markdown")
    for opp in opps:
        await safe_send(update.message.reply_text, format_arb(opp), parse_mode="Markdown")


async def propfirm_setup_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args    = context.args

    if not args:
        account  = load_account(chat_id)
        balance  = account["balance"]
        initial  = account["initial_balance"]
        pnl_pct  = (balance - initial) / initial * 100
        bal   = account["balance"]
        init  = account["initial_balance"]
        tgt   = account["profit_target_pct"]
        ddd   = account["max_daily_dd_pct"]
        tdd   = account["max_total_dd_pct"]
        risk  = account["risk_per_trade_pct"]
        firm  = account["firm"]
        msg_lines = [
            "*🛡️ Prop Firm Setup*",
            "",
            "Current settings:",
            f"• Balance: `${bal:,.2f}`",
            f"• P&L: `{pnl_pct:+.2f}%`",
            f"• Profit Target: `{tgt}%`",
            f"• Max Daily DD: `{ddd}%`",
            f"• Max Total DD: `{tdd}%`",
            f"• Risk/Trade: `{risk}%`",
            f"• Firm: `{firm}`",
            "",
            "*Commands:*",
            "`/propfirm balance 10000` — set balance",
            "`/propfirm target 10` — profit target %",
            "`/propfirm dailydd 5` — daily drawdown %",
            "`/propfirm totaldd 10` — total drawdown %",
            "`/propfirm risk 1` — risk per trade %",
            "`/propfirm firm FTMO` — set firm name",
            "`/propfirm reset` — reset to defaults",
        ]
        await update.message.reply_text(
            "\n".join(msg_lines),
            parse_mode="Markdown",
        )
        return

    account = load_account(chat_id)
    cmd = args[0].lower()

    try:
        if cmd == "balance" and len(args) > 1:
            val = float(args[1])
            account["balance"]             = val
            account["initial_balance"]     = val
            account["daily_start_balance"] = val
            await update.message.reply_text(f"✅ Balance set to `${val:,.2f}`", parse_mode="Markdown")
        elif cmd == "target" and len(args) > 1:
            account["profit_target_pct"] = float(args[1])
            await update.message.reply_text(f"✅ Profit target set to `{args[1]}%`", parse_mode="Markdown")
        elif cmd == "dailydd" and len(args) > 1:
            account["max_daily_dd_pct"] = float(args[1])
            await update.message.reply_text(f"✅ Daily drawdown limit set to `{args[1]}%`", parse_mode="Markdown")
        elif cmd == "totaldd" and len(args) > 1:
            account["max_total_dd_pct"] = float(args[1])
            await update.message.reply_text(f"✅ Total drawdown limit set to `{args[1]}%`", parse_mode="Markdown")
        elif cmd == "risk" and len(args) > 1:
            account["risk_per_trade_pct"] = float(args[1])
            await update.message.reply_text(f"✅ Risk per trade set to `{args[1]}%`", parse_mode="Markdown")
        elif cmd == "firm" and len(args) > 1:
            account["firm"] = args[1]
            await update.message.reply_text(f"✅ Firm set to `{args[1]}`", parse_mode="Markdown")
        elif cmd == "reset":
            account = default_account()
            await update.message.reply_text("✅ Account reset to defaults.", parse_mode="Markdown")
        else:
            await update.message.reply_text("❌ Unknown command. Use /propfirm for help.")
            return
    except ValueError:
        await update.message.reply_text("❌ Invalid value. Use numbers only.")
        return

    account = DrawdownTracker.check_limits(account)
    save_account(chat_id, account)


async def journal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(get_journal_text(), parse_mode="Markdown")


async def logtrade_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    args    = context.args

    if len(args) < 6:
        await update.message.reply_text(
            "Usage: `/logtrade PAIR DIRECTION ENTRY EXIT LOTS RESULT`\n"
            "Example: `/logtrade EURUSD BUY 1.0850 1.0900 0.1 TP`\n"
            "Result: TP or SL",
            parse_mode="Markdown",
        )
        return

    try:
        from signals import PIP_SIZE
        pair      = args[0].upper()
        direction = args[1].upper()
        entry     = float(args[2])
        exit_p    = float(args[3])
        lots      = float(args[4])
        result    = args[5].upper()
        pip_size  = PIP_SIZE.get(pair, 0.0001)

        account = load_account(chat_id)
        account = DrawdownTracker.reset_daily_if_needed(account)
        account = DrawdownTracker.record_trade(
            account, pair, direction, entry, exit_p, lots, pip_size, result
        )
        save_account(chat_id, account)

        pnl_pips = (exit_p - entry) / pip_size if direction == "BUY" \
                   else (entry - exit_p) / pip_size
        pnl_usd  = round(pnl_pips * lots * 10, 2)
        emoji    = "✅" if pnl_usd > 0 else "❌"

        bal_now = account['balance']
        status  = account['status']
        trade_msg = (
            f"{emoji} *Trade Logged*\n\n"
            f"{pair} {direction} | `{pnl_pips:+.1f} pips` | `${pnl_usd:+.2f}`\n"
            f"Balance: `${bal_now:,.2f}`\n"
            f"Status: {status}"
        )
        await update.message.reply_text(trade_msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {e}")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    account = load_account(chat_id)
    account = DrawdownTracker.reset_daily_if_needed(account)
    account = DrawdownTracker.check_limits(account)
    save_account(chat_id, account)

    balance    = account["balance"]
    initial    = account["initial_balance"]
    profit_pct = (balance - initial) / initial * 100
    daily_pnl  = account["daily_pnl"]
    daily_dd   = account["max_daily_dd_pct"]
    total_dd   = account["max_total_dd_pct"]
    target     = account["profit_target_pct"]

    daily_used = max(0, -daily_pnl / account["daily_start_balance"] * 100)
    total_used = max(0, -profit_pct)

    is_wk, wk_msg = NewsFilter.is_weekend()
    events        = await get_news_events()

    status_map = {
        "ACTIVE":      "✅ Active — OK to trade",
        "DAILY_LIMIT": "🛑 Daily limit hit — stop today",
        "TOTAL_LIMIT": "❌ Total drawdown hit — stop trading",
        "TARGET_HIT":  "🎯 Target reached!",
    }

    lines = [
        f"🛡️ *Account Status — {account['firm']}*",
        "",
        f"💰 Balance: `${balance:,.2f}`",
        f"📈 P&L: `{profit_pct:+.2f}%` (target: `{target}%`)",
        f"📊 Status: {status_map.get(account['status'], account['status'])}",
        "",
        f"📅 Daily P&L: `${daily_pnl:+.2f}` | DD used: `{daily_used:.2f}%/{daily_dd}%`",
        f"🛡️ Total DD used: `{total_used:.2f}%/{total_dd}%`",
        "",
        f"🗓️ Weekend: {'🚫 Closed' if is_wk else '✅ Open'}",
        f"📰 News Events Loaded: `{len(events)}`",
    ]

    await safe_send(update.message.reply_text, "\n".join(lines), parse_mode="Markdown")


async def dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(dd_tracker.get_dashboard(), parse_mode="Markdown")


async def setup_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    usage = (
        "⚙️ *Setup Prop Firm*\n\n"
        "Usage: `/setup <firm> <balance> <profit_target> <daily_dd> <total_dd> <risk_pct>`\n\n"
        "Example for FTMO $10k:\n"
        "`/setup FTMO 10000 1000 500 1000 1`\n\n"
        "Parameters:\n"
        "firm — Firm name | balance — Account size\n"
        "profit_target — Target USD | daily_dd — Max daily DD\n"
        "total_dd — Max total DD | risk_pct — Risk per trade %"
    )
    if not context.args or len(context.args) < 6:
        await update.message.reply_text(usage, parse_mode="Markdown")
        return
    try:
        firm     = context.args[0]
        balance  = float(context.args[1])
        profit_t = float(context.args[2])
        daily_dd = float(context.args[3])
        total_dd = float(context.args[4])
        risk_pct = float(context.args[5])
        data = get_data()
        data["account"].update({
            "firm": firm, "balance": balance, "starting": balance,
            "daily_start": balance, "peak": balance,
            "profit_target": profit_t, "max_daily_dd": daily_dd,
            "max_total_dd": total_dd, "risk_per_trade": risk_pct,
        })
        data["trades"] = []; data["trading_days"] = []
        data["daily_pnl"] = 0.0; data["total_pnl"] = 0.0
        data["status"] = "ACTIVE"
        save_data(data)
        msg = (
            f"✅ *Prop Firm Setup Complete*\n\n"
            f"Firm: {firm}\n"
            f"Balance: ${balance:,.2f}\n"
            f"Profit Target: ${profit_t:,.2f}\n"
            f"Max Daily DD: ${daily_dd:,.2f}\n"
            f"Max Total DD: ${total_dd:,.2f}\n"
            f"Risk per Trade: {risk_pct}%\n\n"
            f"Use /dashboard to track progress."
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    except (ValueError, IndexError):
        await update.message.reply_text(usage, parse_mode="Markdown")


async def lotsize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args) < 3:
        await update.message.reply_text(
            "Usage: `/lotsize PAIR ENTRY SL`\nExample: `/lotsize EURUSD 1.0850 1.0830`",
            parse_mode="Markdown"
        )
        return
    try:
        pair   = context.args[0].upper()
        entry  = float(context.args[1])
        sl     = float(context.args[2])
        data   = get_data()
        acc    = data["account"]
        result = calculate_lot_size(pair, entry, sl, acc["balance"], acc["risk_per_trade"])
        lines = [
            f"🧮 *Position Size Calculator*",
            f"",
            f"Pair: *{pair}*",
            f"Entry: `{entry}`",
            f"Stop Loss: `{sl}`",
            f"",
            f"📊 SL Distance: `{result['sl_pips']} pips`",
            f"💰 Risk Amount: `${result['risk_amount']}` ({result['risk_pct']}%)",
            f"📦 *Lot Size: `{result['lot_size']}`*",
            f"",
            f"Account Balance: `${acc['balance']:,.2f}`",
        ]
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except (ValueError, IndexError):
        await update.message.reply_text("❌ Invalid format. Use: /lotsize EURUSD 1.0850 1.0830")


async def updatepnl_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Usage: `/updatepnl 150.50` or `/updatepnl -75.00`",
            parse_mode="Markdown"
        )
        return
    try:
        pnl  = float(context.args[0])
        data = dd_tracker.update_balance(pnl)
        acc  = data["account"]
        emoji = "🟢" if pnl > 0 else "🔴"
        msg = (
            f"{emoji} *Trade Closed*\n\n"
            f"PnL: `${pnl:+.2f}`\n"
            f"New Balance: `${acc['balance']:,.2f}`\n"
            f"Status: {data['status']}\n\n"
            f"Use /dashboard to see full stats."
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    except ValueError:
        await update.message.reply_text("❌ Invalid amount. Use: /updatepnl 150.50")


async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Fetching news...")
    text = await news_engine.get_forex_news()
    await safe_send(msg.edit_text, text, parse_mode="Markdown")


async def subscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if context.job_queue.get_jobs_by_name(f"hourly_{chat_id}"):
        await update.message.reply_text("✅ Already subscribed to hourly signals.")
        return
    context.job_queue.run_repeating(
        auto_signal_job, interval=3600, first=10,
        chat_id=chat_id, name=f"hourly_{chat_id}",
    )
    await update.message.reply_text("✅ Subscribed! Hourly HIGH/MEDIUM signals enabled.")


async def unsubscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    jobs = context.job_queue.get_jobs_by_name(f"hourly_{chat_id}")
    if not jobs:
        await update.message.reply_text("❌ Not subscribed to hourly signals.")
        return
    for job in jobs:
        job.schedule_removal()
    await update.message.reply_text("🔕 Unsubscribed from hourly signals.")


async def subscribe_bestpicks_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id  = update.effective_chat.id
    job_name = f"bestpicks_{chat_id}"

    try:
        hours = int(context.args[0]) if context.args else 4
        hours = max(1, min(hours, 24))
    except (ValueError, IndexError):
        hours = 4

    for job in context.job_queue.get_jobs_by_name(job_name):
        job.schedule_removal()

    context.job_queue.run_repeating(
        best_picks_job,
        interval=hours * 3600,
        first=30,
        chat_id=chat_id,
        name=job_name,
        data={"hours": hours},
    )
    await update.message.reply_text(
        f"✅ *Best Picks Subscribed!*\n\n"
        f"You'll receive the top 3 highest confidence signals every *{hours} hour(s)*.\n"
        f"Only HIGH and MEDIUM quality signals are included.\n\n"
        "Use /unsubscribe_bestpicks to stop.",
        parse_mode="Markdown",
    )


async def unsubscribe_bestpicks_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    jobs    = context.job_queue.get_jobs_by_name(f"bestpicks_{chat_id}")
    if not jobs:
        await update.message.reply_text("❌ Not subscribed to best picks.")
        return
    for job in jobs:
        job.schedule_removal()
    await update.message.reply_text("🔕 Unsubscribed from best picks.")


# ── Callback Buttons ──────────────────────────────────────────────────────────

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    await query.answer()
    data    = query.data
    chat_id = query.message.chat_id

    def send(text, **kw):
        return context.bot.send_message(chat_id=chat_id, text=text, **kw)

    if data == "back_main":
        await context.bot.send_message(
            chat_id=chat_id,
            text="🤖 *Main Menu* — Select an option:",
            parse_mode="Markdown",
            reply_markup=main_keyboard()
        )

    elif data == "menu_majors":
        await context.bot.send_message(
            chat_id=chat_id,
            text="💱 *Major Pairs* — Select a pair:",
            parse_mode="Markdown",
            reply_markup=majors_keyboard()
        )

    elif data == "menu_minors":
        await context.bot.send_message(
            chat_id=chat_id,
            text="🔀 *Minor Pairs* — Select a pair:",
            parse_mode="Markdown",
            reply_markup=minors_keyboard()
        )

    elif data == "menu_commodities":
        await context.bot.send_message(
            chat_id=chat_id,
            text="🥇 *Commodities* — Select a pair:",
            parse_mode="Markdown",
            reply_markup=commodities_keyboard()
        )

    elif data == "menu_scalp":
        await context.bot.send_message(
            chat_id=chat_id,
            text="⚡ *Scalp Signal* — Select pair:",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("EUR/USD", callback_data="scalp_EURUSD"),
                 InlineKeyboardButton("GBP/USD", callback_data="scalp_GBPUSD")],
                [InlineKeyboardButton("USD/JPY", callback_data="scalp_USDJPY"),
                 InlineKeyboardButton("XAU/USD", callback_data="scalp_XAUUSD")],
                [InlineKeyboardButton("AUD/USD", callback_data="scalp_AUDUSD"),
                 InlineKeyboardButton("GBP/JPY", callback_data="scalp_GBPJPY")],
                [InlineKeyboardButton("🔙 Back",  callback_data="back_main")],
            ])
        )

    elif data == "menu_propfirm":
        await context.bot.send_message(
            chat_id=chat_id,
            text="🛡️ *Prop Firm Tools*",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 Dashboard",    callback_data="dashboard")],
                [InlineKeyboardButton("📓 Journal",      callback_data="journal")],
                [InlineKeyboardButton("🔙 Back",         callback_data="back_main")],
            ])
        )

    elif data == "signals_all":
        await send("⏳ Scanning all 25 pairs...")
        await send_signals(send, PAIRS)

    elif data == "signals_majors":
        await send("⏳ Scanning major pairs...")
        await send_signals(send, MAJORS)

    elif data == "signals_minors":
        await send("⏳ Scanning minor pairs...")
        await send_signals(send, MINORS)

    elif data == "signals_commodities":
        await send("⏳ Scanning commodities...")
        await send_signals(send, COMMODITIES)

    elif data.startswith("signal_"):
        pair = data.replace("signal_", "")
        await send(f"⏳ Analyzing {pair}...")
        pf_data = get_data()
        limits  = dd_tracker.check_limits(pf_data)
        if not limits["allowed"]:
            await send("🛡️ *Trading Halted*\n\n" + limits["reason"], parse_mode="Markdown")
            return
        blackout = await get_news_blackout(pair)
        if blackout["blocked"]:
            await send(
                "📰 *News Blackout*\n\n" + blackout["reason"] + "\nResumes: " + blackout["next_clear"],
                parse_mode="Markdown"
            )
            return
        result = await signal_engine.get_signal(pair)
        if result["direction"] != "N/A":
            acc = pf_data["account"]
            ls  = calculate_lot_size(pair, result["entry"], result["sl"],
                                     acc["balance"], acc["risk_per_trade"])
            result["lot_size"]    = ls["lot_size"]
            result["risk_amount"] = ls["risk_amount"]
            result["sl_pips"]     = ls["sl_pips"]
            log_signal(pair, result["direction"], result["entry"], result["sl"],
                       result["tp1"], result["tp2"], result["confidence"], ls["lot_size"])
        await safe_send(send, format_signal(result), parse_mode="Markdown")

    elif data.startswith("scalp_"):
        pair = data.replace("scalp_", "")
        await send(f"⚡ Analyzing {pair} M5+M15 scalp...")
        result = await scalp_engine.get_scalp_signal(pair)
        await safe_send(send, format_scalp(result), parse_mode="Markdown")

    elif data == "arb":
        await send("🔄 Scanning arbitrage opportunities...")
        opps = await arb_engine.scan_all()
        if not opps:
            await send("🔄 No arbitrage opportunities right now. Z-score within normal range. Try again in 30-60 min.")
        else:
            await send(f"🔄 *{len(opps)} Arbitrage Signal(s) Found*", parse_mode="Markdown")
            for opp in opps:
                await safe_send(send, format_arb(opp), parse_mode="Markdown")

    elif data == "bestpicks":
        await send("⏳ Scanning all pairs for best picks...")
        picks = await get_best_picks(PAIRS)
        if not picks:
            await send("😴 No HIGH/MEDIUM signals right now. Try again later.")
            return
        await send(f"🏆 *Top {len(picks)} Best Picks Right Now*", parse_mode="Markdown")
        for i, pick in enumerate(picks, 1):
            await safe_send(send, format_bestpick(pick, i), parse_mode="Markdown")

    elif data == "dashboard":
        await safe_send(send, dd_tracker.get_dashboard(), parse_mode="Markdown")

    elif data == "journal":
        await safe_send(send, get_journal_text(), parse_mode="Markdown")

    elif data == "news":
        await send("⏳ Fetching news...")
        text = await news_engine.get_forex_news()
        await safe_send(send, text, parse_mode="Markdown")

    elif data == "help":
        await send(
            "⭐⭐⭐ HIGH   → H4 aligned + London/NY session\n"
            "⭐⭐ MEDIUM  → H4 aligned\n"
            "⚠️ LOW      → H4 conflict — skip\n\n"
            "/subscribe_bestpicks 4 → best picks every 4hrs\n"
            "/setup FTMO 10000 1000 500 1000 1 → prop firm setup\n"
            "Trade HIGH quality signals only."
        )

    else:
        await send(f"Unknown action: {data}")


# ── Background Jobs ───────────────────────────────────────────────────────────

async def auto_signal_job(context: ContextTypes.DEFAULT_TYPE):
    """Hourly job: send best picks to subscribed users."""
    picks = await get_best_picks(PAIRS)
    if not picks:
        return
    await context.bot.send_message(
        chat_id=context.job.chat_id,
        text="🕐 *Hourly Best Picks*",
        parse_mode="Markdown"
    )
    for i, pick in enumerate(picks, 1):
        await context.bot.send_message(
            chat_id=context.job.chat_id,
            text=format_bestpick(pick, i),
            parse_mode="Markdown"
        )


async def best_picks_job(context: ContextTypes.DEFAULT_TYPE):
    """Scheduled best picks job (every N hours)."""
    hours = context.job.data.get("hours", 4)
    picks = await get_best_picks(PAIRS)
    if not picks:
        await context.bot.send_message(
            chat_id=context.job.chat_id,
            text="😴 No HIGH/MEDIUM signals right now. Try again later.",
        )
        return
    await context.bot.send_message(
        chat_id=context.job.chat_id,
        text=f"🏆 *Best Picks (every {hours}h)*",
        parse_mode="Markdown"
    )
    for i, pick in enumerate(picks, 1):
        await context.bot.send_message(
            chat_id=context.job.chat_id,
            text=format_bestpick(pick, i),
            parse_mode="Markdown"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    app = (
        Application.builder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .build()
    )

    app.add_handler(CommandHandler("start",                 start))
    app.add_handler(CommandHandler("help",                  help_command))
    app.add_handler(CommandHandler("signal",                signal_command))
    app.add_handler(CommandHandler("signals",               signals_command))
    app.add_handler(CommandHandler("majors",                majors_command))
    app.add_handler(CommandHandler("minors",                minors_command))
    app.add_handler(CommandHandler("commodities",           commodities_command))
    app.add_handler(CommandHandler("bestpicks",             bestpicks_command))
    app.add_handler(CommandHandler("scalp",                 scalp_command))
    app.add_handler(CommandHandler("arb",                   arb_command))
    app.add_handler(CommandHandler("news",                  news_command))
    app.add_handler(CommandHandler("subscribe",             subscribe_command))
    app.add_handler(CommandHandler("unsubscribe",           unsubscribe_command))
    app.add_handler(CommandHandler("subscribe_bestpicks",   subscribe_bestpicks_command))
    app.add_handler(CommandHandler("unsubscribe_bestpicks", unsubscribe_bestpicks_command))
    app.add_handler(CommandHandler("dashboard",             dashboard_command))
    app.add_handler(CommandHandler("setup",                 setup_command))
    app.add_handler(CommandHandler("propfirm",              propfirm_setup_command))
    app.add_handler(CommandHandler("lotsize",               lotsize_command))
    app.add_handler(CommandHandler("updatepnl",             updatepnl_command))
    app.add_handler(CommandHandler("journal",               journal_command))
    app.add_handler(CommandHandler("logtrade",              logtrade_command))
    app.add_handler(CommandHandler("status",                status_command))

    app.add_handler(CallbackQueryHandler(button_handler))

    logger.info("🤖 Bot started. Polling...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
