import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TWELVEDATA_API_KEY: str = os.getenv("TWELVEDATA_API_KEY", "")

    def __init__(self):
        if not self.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN is not set")
        if not self.TWELVEDATA_API_KEY:
            raise ValueError("TWELVEDATA_API_KEY is not set")
