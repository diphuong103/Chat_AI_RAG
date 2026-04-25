"""
VietMoney — Exchange Rate Service
Fetches live foreign exchange rates from public APIs.
Provides formatted rate info for the chatbot prompt.
"""

import logging
import time
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Cache duration in seconds (5 minutes)
CACHE_TTL = 300

# Primary API: exchangerate-api.com (free tier, no key needed)
EXCHANGE_API_URL = "https://open.er-api.com/v6/latest/USD"

# Currencies relevant to VietMoney
TARGET_CURRENCIES = [
    "VND", "EUR", "GBP", "JPY", "CNY", "KRW",
    "THB", "SGD", "AUD", "CAD", "CHF", "HKD", "TWD",
]


class ExchangeRateService:
    """Fetches and caches live exchange rates for VietMoney chatbot."""

    def __init__(self):
        self._cache: Optional[dict] = None
        self._cache_time: float = 0

    def _fetch_rates(self) -> Optional[dict]:
        """Fetch latest rates from the exchange rate API."""
        try:
            resp = requests.get(EXCHANGE_API_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("result") == "success":
                rates = data.get("rates", {})
                logger.info(f"Fetched {len(rates)} exchange rates (base=USD)")
                return rates
            else:
                logger.warning(f"API returned non-success: {data.get('result')}")
                return None
        except requests.RequestException as e:
            logger.error(f"Failed to fetch exchange rates: {e}")
            return None

    def get_rates(self) -> Optional[dict]:
        """Get exchange rates (with caching)."""
        now = time.time()
        if self._cache and (now - self._cache_time) < CACHE_TTL:
            return self._cache

        rates = self._fetch_rates()
        if rates:
            self._cache = rates
            self._cache_time = now
        return self._cache  # Return old cache if fetch failed

    def format_rates_for_prompt(self) -> str:
        """Format exchange rates as a readable string for the LLM prompt.

        Returns:
            Formatted string with key exchange rates, or empty string if unavailable.
        """
        rates = self.get_rates()
        if not rates:
            return ""

        vnd_rate = rates.get("VND")
        if not vnd_rate:
            return ""

        lines = []
        lines.append("📊 TỶ GIÁ NGOẠI TỆ (Cập nhật real-time):")
        lines.append(f"   Thời điểm cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        lines.append("")
        lines.append("   Đơn vị: 1 ngoại tệ = ? VND")
        lines.append("   ─────────────────────────────────")

        # USD → VND (base rate)
        usd_to_vnd = vnd_rate
        lines.append(f"   💵 USD (Đô la Mỹ):        {usd_to_vnd:,.0f} VND")

        # Other currencies → VND (cross rate via USD)
        currency_names = {
            "EUR": "Euro",
            "GBP": "Bảng Anh",
            "JPY": "Yên Nhật (100¥)",
            "CNY": "Nhân dân tệ",
            "KRW": "Won Hàn Quốc (1000₩)",
            "THB": "Bạt Thái",
            "SGD": "Đô la Singapore",
            "AUD": "Đô la Úc",
            "CAD": "Đô la Canada",
            "CHF": "Franc Thụy Sĩ",
            "HKD": "Đô la Hồng Kông",
            "TWD": "Đô la Đài Loan",
        }

        currency_icons = {
            "EUR": "💶", "GBP": "💷", "JPY": "💴", "CNY": "🇨🇳",
            "KRW": "🇰🇷", "THB": "🇹🇭", "SGD": "🇸🇬", "AUD": "🇦🇺",
            "CAD": "🇨🇦", "CHF": "🇨🇭", "HKD": "🇭🇰", "TWD": "🇹🇼",
        }

        for ccy in TARGET_CURRENCIES:
            if ccy == "VND":
                continue
            if ccy == "USD":
                continue

            ccy_rate = rates.get(ccy)
            if not ccy_rate or ccy_rate == 0:
                continue

            # Cross rate: 1 CCY = (VND_rate / CCY_rate) VND
            cross = usd_to_vnd / ccy_rate

            icon = currency_icons.get(ccy, "💱")
            name = currency_names.get(ccy, ccy)

            # Special formatting for JPY (per 100) and KRW (per 1000)
            if ccy == "JPY":
                cross_100 = cross * 100
                lines.append(f"   {icon} {ccy} ({name}):  {cross_100:,.0f} VND")
            elif ccy == "KRW":
                cross_1000 = cross * 1000
                lines.append(f"   {icon} {ccy} ({name}): {cross_1000:,.0f} VND")
            else:
                lines.append(f"   {icon} {ccy} ({name}):  {cross:,.0f} VND")

        lines.append("")
        lines.append("   ⚠️ Lưu ý: Tỷ giá trên là tỷ giá tham khảo thị trường.")
        lines.append("   Tỷ giá mua/bán thực tế tại VietMoney có thể chênh lệch nhỏ.")

        return "\n".join(lines)
