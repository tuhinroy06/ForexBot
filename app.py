"""
Forex Signal Telegram Bot v2
"""
import asyncio
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from signals import SignalEngine
from news import NewsEngine
from config import Config

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config()
signal_engine = SignalEngine(config.ALPHA_VANTAGE_API_KEY)
news_engine = NewsEngine()
PAIRS = [
    # Majors
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    # Minors
    "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY",
    "EURCHF", "EURAUD", "EURCAD", "GBPAUD", "GBPCAD", "GBPCHF",
    "AUDCAD", "AUDCHF", "AUDNZD", "NZDJPY",
    # Commodities
    "XAUUSD", "XAGUSD",
]

MAJORS    = ["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD"]
MINORS    = ["EURGBP","EURJPY","GBPJPY","AUDJPY","CADJPY","CHFJPY",
             "EURCHF","EURAUD","EURCAD","GBPAUD","GBPCAD","GBPCHF",
             "AUDCAD","AUDCHF","AUDNZD","NZDJPY"]
COMMODITIES = ["XAUUSD","XAGUSD"]

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 All Signals", callback_data="signals_all")],
        [InlineKeyboardButton("💱 Majors",      callback_data="signals_majors"),
         InlineKeyboardButton("🔀 Minors",      callback_data="signals_minors")],
        [InlineKeyboardButton("🥇 Commodities", callback_data="signals_commodities")],
        [InlineKeyboardButton("📰 News",        callback_data="news"),
         InlineKeyboardButton("ℹ️ Help",        callback_data="help")],
    ])

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Forex Signal Bot v2*\n\n"
        "Enhanced accuracy via:\n"
        "• H1 + H4 multi-timeframe confirmation\n"
        "• London/NY session filter\n"
        "• RSI, MACD, EMA, BB, Stoch, ADX, ATR\n"
        "• ML-based confidence scoring\n"
        "• ATR-dynamic SL/TP levels\n\n"
        "Select below:",
        parse_mode="Markdown", reply_markup=main_keyboard())

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *Commands*\n\n"
        "/start — Main menu\n"
        "/signal EURUSD — Single pair\n"
        "/signals — All 4 pairs\n"
        "/news — Economic calendar\n"
        "/subscribe — Hourly auto-signals\n"
        "/unsubscribe — Stop\n\n"
        "⭐ Signal Quality:\n"
        "`⭐⭐⭐ HIGH` — H4 aligned + in session\n"
        "`⭐⭐ MEDIUM` — H4 aligned\n"
        "`⚠️ LOW` — H4 conflict, skip",
        parse_mode="Markdown")

async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /signal EURUSD")
        return
    pair = context.args[0].upper()
    if pair not in PAIRS:
        await update.message.reply_text("❌ Use: EURUSD GBPUSD USDJPY XAUUSD")
        return
    msg = await update.message.reply_text("⏳ Analyzing H1 + H4...")
    result = await signal_engine.get_signal(pair)
    await msg.edit_text(format_signal(result), parse_mode="Markdown")

async def signals_all_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Scanning all pairs...")
    parts = [format_signal(await signal_engine.get_signal(p)) for p in PAIRS]
    await msg.edit_text("\n\n─────────────────\n\n".join(parts), parse_mode="Markdown")

async def news_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("⏳ Fetching news...")
    await msg.edit_text(await news_engine.get_forex_news(), parse_mode="Markdown")

async def subscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if context.job_queue.get_jobs_by_name(str(chat_id)):
        await update.message.reply_text("✅ Already subscribed.")
        return
    context.job_queue.run_repeating(auto_signal_job, interval=3600, first=10,
                                    chat_id=chat_id, name=str(chat_id))
    await update.message.reply_text("✅ Subscribed! Hourly HIGH/MEDIUM signals only.")

async def unsubscribe_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    jobs = context.job_queue.get_jobs_by_name(str(update.effective_chat.id))
    if not jobs:
        await update.message.reply_text("❌ Not subscribed.")
        return
    for job in jobs: job.schedule_removal()
    await update.message.reply_text("🔕 Unsubscribed.")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data == "signals_all":
        await query.edit_message_text("⏳ Scanning all pairs...")
        parts = [format_signal(await signal_engine.get_signal(p)) for p in PAIRS]
        await query.edit_message_text("\n\n─────────────────\n\n".join(parts), parse_mode="Markdown")
    elif data.startswith("signal_"):
        pair = data.replace("signal_", "")
        await query.edit_message_text("⏳ Analyzing H1 + H4...")
        result = await signal_engine.get_signal(pair)
        await query.edit_message_text(format_signal(result), parse_mode="Markdown")
    elif data == "news":
        await query.edit_message_text("⏳ Fetching news...")
        await query.edit_message_text(await news_engine.get_forex_news(), parse_mode="Markdown")
    elif data == "help":
        await query.edit_message_text(
            "⭐⭐⭐ HIGH  → H4 aligned + London/NY + conf ≥75%\n"
            "⭐⭐ MEDIUM → H4 aligned\n"
            "⚠️ LOW     → H4 conflict — skip\n\n"
            "Best results: trade HIGH quality only.", parse_mode="Markdown")

async def auto_signal_job(context: ContextTypes.DEFAULT_TYPE):
    parts = []
    for pair in PAIRS:
        result = await signal_engine.get_signal(pair)
        if "LOW" not in result.get("quality", ""):
            parts.append(format_signal(result))
    text = ("🔔 *Hourly Signal Update*\n\n" + "\n\n─────────────────\n\n".join(parts)
            if parts else "🔔 No HIGH/MEDIUM signals right now. Market off-session or ranging.")
    await context.bot.send_message(chat_id=context.job.chat_id, text=text, parse_mode="Markdown")

def format_signal(s: dict) -> str:
    dir_emoji = "🟢 BUY" if s["direction"] == "BUY" else ("🔴 SELL" if s["direction"] == "SELL" else "⚪ N/A")
    conf = s["confidence"]
    bar = "█" * int(conf / 10) + "░" * (10 - int(conf / 10))
    ml_tag = " _(ML)_" if s.get("ml_used") else ""
    sess_emoji = "✅" if s.get("in_session") else "⏸"
    ind = s.get("indicators", {})
    return (
        f"*{s['pair']}* — {dir_emoji}\n"
        f"🕐 {s['timestamp']}\n"
        f"📋 `{s['timeframe']}`\n\n"
        f"💰 *Entry:* `{s['entry']}`\n"
        f"🛑 *SL:* `{s['sl']}`\n"
        f"🎯 *TP1:* `{s['tp1']}`\n"
        f"🎯 *TP2:* `{s['tp2']}`\n\n"
        f"📊 *Confidence:* {conf}%{ml_tag} `{bar}`\n"
        f"🏆 *Quality:* {s.get('quality','N/A')}\n"
        f"📈 *H4 Trend:* {s.get('h4_trend','N/A')}\n"
        f"{sess_emoji} *Session:* {s.get('session','N/A')}\n\n"
        f"*Indicators (H1):*\n"
        f"  • RSI: `{ind.get('rsi',0):.1f}` {ind.get('rsi_signal','')}\n"
        f"  • MACD: {ind.get('macd_signal','')}\n"
        f"  • EMA: {ind.get('ema_cross','')}\n"
        f"  • Stoch: {ind.get('stochastic','')}\n"
        f"  • ADX: {ind.get('adx','')}\n"
        f"  • ATR: `{ind.get('atr',0)}`\n"
        f"  • BB: {ind.get('bb_position','')}\n\n"
        f"📰 *News:* {s.get('news_sentiment','N/A')}\n"
        f"⚠️ _Trade HIGH quality only. Manage risk properly._"
    )

def main():
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    for cmd, fn in [("start", start), ("help", help_command), ("signal", signal_command),
                    ("signals", signals_all_command), ("news", news_command),
                    ("subscribe", subscribe_command), ("unsubscribe", unsubscribe_command)]:
        app.add_handler(CommandHandler(cmd, fn))
    app.add_handler(CallbackQueryHandler(button_handler))
    logger.info("Bot v2 started.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
