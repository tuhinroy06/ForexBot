"""
Forex Signal Bot v2 — app.py
Fixed: callbacks use send_message (no edit) to avoid timeout & BadRequest errors
Fixed: Message_too_long — each signal sent as separate message
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

config      = Config()
signal_engine = SignalEngine(config.ALPHA_VANTAGE_API_KEY)
news_engine = NewsEngine()

MAJORS      = ["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD"]
MINORS      = ["EURGBP","EURJPY","GBPJPY","AUDJPY","CADJPY","CHFJPY",
               "EURCHF","EURAUD","EURCAD","GBPAUD","GBPCAD","GBPCHF",
               "AUDCAD","AUDCHF","AUDNZD","NZDJPY"]
COMMODITIES = ["XAUUSD","XAGUSD"]
PAIRS       = MAJORS + MINORS + COMMODITIES


# ── Formatter ─────────────────────────────────────────────────────────────────

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


# ── Safe sender (splits messages > 4096 chars) ────────────────────────────────

async def safe_send(send_fn, text: str, **kwargs):
    if len(text) <= 4096:
        await send_fn(text, **kwargs)
    else:
        for i in range(0, len(text), 4096):
            await send_fn(text[i:i+4096], **kwargs)


# ── Core signal sender ────────────────────────────────────────────────────────

async def send_signals(send_fn, pairs: list):
    """Send one message per pair. Adds delay to respect Alpha Vantage rate limit (5 req/min)."""
    for i, pair in enumerate(pairs):
        if i > 0:
            # AV free tier: 5 req/min. Each signal uses 2 calls (H1+H4).
            # 13s gap keeps us safely under the limit.
            await send_fn(f"⏳ Fetching {pair}... ({i+1}/{len(pairs)})")
            await asyncio.sleep(13)
        result = await signal_engine.get_signal(pair)
        text = format_signal(result)
        await safe_send(send_fn, text, parse_mode="Markdown")


# ── Commands ──────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 All Signals",  callback_data="signals_all")],
        [InlineKeyboardButton("💱 Majors",        callback_data="signals_majors"),
         InlineKeyboardButton("🔀 Minors",        callback_data="signals_minors")],
        [InlineKeyboardButton("🥇 Commodities",   callback_data="signals_commodities")],
        [InlineKeyboardButton("📰 News",          callback_data="news"),
         InlineKeyboardButton("ℹ️ Help",          callback_data="help")],
    ])
    await update.message.reply_text(
        "🤖 *Forex Signal Bot v2*\n\n"
        "✅ H1 + H4 multi-timeframe\n"
        "✅ London / NY session filter\n"
        "✅ RSI, MACD, EMA, BB, Stoch, ADX, ATR\n"
        "✅ ML confidence scoring\n"
        "✅ 25 currency pairs\n\n"
        "Select an option:",
        parse_mode="Markdown",
        reply_markup=keyboard,
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
        "/news — Economic calendar\n"
        "/subscribe — Hourly auto signals\n"
        "/unsubscribe — Stop\n\n"
        "⭐ Quality:\n"
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

async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Fetching news...")
    text = await news_engine.get_forex_news()
    await safe_send(msg.edit_text, text, parse_mode="Markdown")

async def subscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if context.job_queue.get_jobs_by_name(str(chat_id)):
        await update.message.reply_text("✅ Already subscribed.")
        return
    context.job_queue.run_repeating(
        auto_signal_job, interval=3600, first=10,
        chat_id=chat_id, name=str(chat_id),
    )
    await update.message.reply_text("✅ Subscribed! Hourly HIGH/MEDIUM signals enabled.")

async def unsubscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    jobs = context.job_queue.get_jobs_by_name(str(update.effective_chat.id))
    if not jobs:
        await update.message.reply_text("❌ Not subscribed.")
        return
    for job in jobs:
        job.schedule_removal()
    await update.message.reply_text("🔕 Unsubscribed.")


# ── Callback Handler (uses send_message — NOT edit — to avoid timeouts) ───────

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    await query.answer()                        # stops spinner immediately
    data    = query.data
    chat_id = query.message.chat_id

    # Always send NEW messages, never edit existing ones
    # edit_message_text causes BadRequest when API calls take > a few seconds
    def send(text, **kw):
        return context.bot.send_message(chat_id=chat_id, text=text, **kw)

    if data == "signals_all":
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
        result = await signal_engine.get_signal(pair)
        await safe_send(send, format_signal(result), parse_mode="Markdown")

    elif data == "news":
        await send("⏳ Fetching news...")
        text = await news_engine.get_forex_news()
        await safe_send(send, text, parse_mode="Markdown")

    elif data == "help":
        await send(
            "⭐⭐⭐ HIGH   → H4 aligned + London/NY session\n"
            "⭐⭐ MEDIUM  → H4 aligned, may be off-session\n"
            "⚠️ LOW      → H4 conflict — skip\n\n"
            "Trade HIGH quality signals only."
        )


# ── Auto Hourly Job ───────────────────────────────────────────────────────────

async def auto_signal_job(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.chat_id

    def send(text, **kw):
        return context.bot.send_message(chat_id=chat_id, text=text, **kw)

    signals = []
    for pair in PAIRS:
        result = await signal_engine.get_signal(pair)
        if "LOW" not in result.get("quality", ""):
            signals.append(result)

    if not signals:
        await send("🔔 No HIGH/MEDIUM signals this hour. Market off-session or ranging.")
        return

    await send(f"🔔 *Hourly Update — {len(signals)} signals*", parse_mode="Markdown")
    for result in signals:
        await safe_send(send, format_signal(result), parse_mode="Markdown")


# ── Error Handler ─────────────────────────────────────────────────────────────

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Error: {context.error}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start",       start))
    app.add_handler(CommandHandler("help",        help_command))
    app.add_handler(CommandHandler("signal",      signal_command))
    app.add_handler(CommandHandler("signals",     signals_command))
    app.add_handler(CommandHandler("majors",      majors_command))
    app.add_handler(CommandHandler("minors",      minors_command))
    app.add_handler(CommandHandler("commodities", commodities_command))
    app.add_handler(CommandHandler("news",        news_command))
    app.add_handler(CommandHandler("subscribe",   subscribe_command))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_error_handler(error_handler)

    logger.info("Bot started.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
