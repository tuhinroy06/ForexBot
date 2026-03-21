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
from signals import SignalEngine
from news import NewsEngine
from config import Config

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

config        = Config()
signal_engine = SignalEngine(config.TWELVEDATA_API_KEY)
news_engine   = NewsEngine()

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


# ── Keyboards ─────────────────────────────────────────────────────────────────

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 All Signals",    callback_data="signals_all")],
        [InlineKeyboardButton("💱 Majors",          callback_data="menu_majors"),
         InlineKeyboardButton("🔀 Minors",          callback_data="menu_minors")],
        [InlineKeyboardButton("🥇 Commodities",     callback_data="menu_commodities")],
        [InlineKeyboardButton("🏆 Best Picks Now",  callback_data="bestpicks")],
        [InlineKeyboardButton("📰 News",            callback_data="news"),
         InlineKeyboardButton("ℹ️ Help",            callback_data="help")],
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
        "/subscribe\\_bestpicks 4 — Auto best picks every 4 hrs\n"
        "/unsubscribe\\_bestpicks — Stop auto best picks\n"
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
    msg = await update.message.reply_text(f"⏳ Analyzing {pair}...")
    result = await signal_engine.get_signal(pair)
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

    # Parse interval from args e.g. /subscribe_bestpicks 4
    try:
        hours = int(context.args[0]) if context.args else 4
        hours = max(1, min(hours, 24))  # clamp between 1-24
    except (ValueError, IndexError):
        hours = 4

    # Remove existing job if any
    for job in context.job_queue.get_jobs_by_name(job_name):
        job.schedule_removal()

    context.job_queue.run_repeating(
        best_picks_job,
        interval=hours * 3600,
        first=30,  # first run 30 seconds after subscribing
        chat_id=chat_id,
        name=job_name,
        data={"hours": hours},
    )
    await update.message.reply_text(
        f"✅ *Best Picks Subscribed!*\n\n"
        f"You'll receive the top 3 highest confidence signals every *{hours} hour(s)*.\n"
        f"Only HIGH and MEDIUM quality signals are included.\n\n"
        f"Use /unsubscribe\\_bestpicks to stop.",
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

    elif data == "bestpicks":
        await send("⏳ Scanning all pairs for best picks...")
        picks = await get_best_picks(PAIRS)
        if not picks:
            await send("😴 No HIGH/MEDIUM signals right now. Try again later.")
            return
        await send(f"🏆 *Top {len(picks)} Best Picks Right Now*", parse_mode="Markdown")
        for i, pick in enumerate(picks, 1):
            await safe_send(send, format_bestpick(pick, i), parse_mode="Markdown")

    elif data.startswith("signal_"):
        pair = data.replace("signal_", "")
        await send(f"⏳ Analyzing {pair}...")
        result = await signal_engine.get_signal(pair)
        await safe_send(send, format_signal(result), parse_mode="Markdown")

    elif data == "news":
        await send("⏳ Fetching news...")
        text = await news_engine.get_forex_news()
        await safe_send(send, text, parse_mode="Markdown")

    elif data == "help":
        await send(
            "⭐⭐⭐ HIGH   → H4 aligned + London/NY session\n"
            "⭐⭐ MEDIUM  → H4 aligned\n"
            "⚠️ LOW      → H4 conflict — skip\n\n"
            "/subscribe\\_bestpicks 4 → auto best picks every 4 hrs\n"
            "Trade HIGH quality signals only."
        )


# ── Scheduled Jobs ────────────────────────────────────────────────────────────

async def auto_signal_job(context: ContextTypes.DEFAULT_TYPE):
    """Hourly job — sends all HIGH/MEDIUM signals."""
    chat_id = context.job.chat_id

    def send(text, **kw):
        return context.bot.send_message(chat_id=chat_id, text=text, **kw)

    signals = []
    for pair in PAIRS:
        result = await signal_engine.get_signal(pair)
        if "LOW" not in result.get("quality", "") and result["direction"] != "N/A":
            signals.append(result)

    if not signals:
        await send("🔔 No HIGH/MEDIUM signals this hour. Market off-session or ranging.")
        return

    await send(f"🔔 *Hourly Update — {len(signals)} signals*", parse_mode="Markdown")
    for result in signals:
        await safe_send(send, format_signal(result), parse_mode="Markdown")


async def best_picks_job(context: ContextTypes.DEFAULT_TYPE):
    """Best picks job — sends top 3 highest confidence signals."""
    chat_id = context.job.chat_id
    hours   = context.job.data.get("hours", 4) if context.job.data else 4

    def send(text, **kw):
        return context.bot.send_message(chat_id=chat_id, text=text, **kw)

    await send("🔍 Scanning all 25 pairs for best picks...")
    picks = await get_best_picks(PAIRS, top_n=3)

    if not picks:
        await send(
            "😴 *No Best Picks Available*\n\n"
            "No HIGH/MEDIUM quality signals found.\n"
            "Market is likely off-session or ranging.\n"
            f"Next scan in {hours} hour(s).",
            parse_mode="Markdown"
        )
        return

    await send(
        f"🏆 *Best Picks — Top {len(picks)} Signals*\n"
        f"📅 {picks[0]['timestamp']}\n"
        f"Next update in {hours} hour(s).",
        parse_mode="Markdown"
    )
    for i, pick in enumerate(picks, 1):
        await safe_send(send, format_bestpick(pick, i), parse_mode="Markdown")


# ── Error Handler ─────────────────────────────────────────────────────────────

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Error: {context.error}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start",                   start))
    app.add_handler(CommandHandler("help",                    help_command))
    app.add_handler(CommandHandler("signal",                  signal_command))
    app.add_handler(CommandHandler("signals",                 signals_command))
    app.add_handler(CommandHandler("majors",                  majors_command))
    app.add_handler(CommandHandler("minors",                  minors_command))
    app.add_handler(CommandHandler("commodities",             commodities_command))
    app.add_handler(CommandHandler("bestpicks",               bestpicks_command))
    app.add_handler(CommandHandler("subscribe_bestpicks",     subscribe_bestpicks_command))
    app.add_handler(CommandHandler("unsubscribe_bestpicks",   unsubscribe_bestpicks_command))
    app.add_handler(CommandHandler("subscribe",               subscribe_command))
    app.add_handler(CommandHandler("unsubscribe",             unsubscribe_command))
    app.add_handler(CommandHandler("news",                    news_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_error_handler(error_handler)

    logger.info("Bot started. Connected to Telegram. Waiting for messages...")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
