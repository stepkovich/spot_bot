import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext, InvalidOperation
from typing import Dict, List, Optional, Any, Set


# ===================================================================
# Rate Limiter — prevents Binance API bans
# ===================================================================

class OrderRateLimiter:
    """Rate limiter for Binance order operations (place / cancel).

    Binance Spot limits:
      - 10 orders/second per UID (WS API + REST share the same counter)
      - 200,000 orders/24h
      - IP weight: 1200/min

    We use 8 orders/sec by default (2/sec safety margin).

    Implementation: simple mutex + next-allowed-timestamp.
    Each acquire() call:
      1. Locks the mutex
      2. Sleeps if next_allowed is in the future
      3. Advances next_allowed by min_interval
      4. Releases mutex

    This serialises all order operations with guaranteed spacing.
    With 30 concurrent gather tasks, total time = 30 × 125ms ≈ 3.75s.
    """

    def __init__(self, orders_per_second: float = 8.0):
        self._min_interval = 1.0 / orders_per_second   # 0.125s at 8/sec
        self._next_allowed = 0.0                        # monotonic timestamp
        self._lock = asyncio.Lock()
        self._total_waits = 0                            # статистика
        self._total_wait_time = 0.0                      # секунд суммарно

    async def acquire(self) -> None:
        """Wait until an order operation slot is available."""
        async with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                self._total_waits += 1
                self._total_wait_time += wait
                if self._total_waits <= 5 or self._total_waits % 50 == 0:
                    logger.debug(
                        f"RateLimiter: waiting {wait:.3f}s "
                        f"(total_waits={self._total_waits})"
                    )
                await asyncio.sleep(wait)
            self._next_allowed = max(now, self._next_allowed) + self._min_interval

    def stats(self) -> str:
        """Return rate limiter statistics for logging."""
        return (
            f"waits={self._total_waits} "
            f"total_wait={self._total_wait_time:.1f}s "
            f"rate={1.0/self._min_interval:.0f}/s"
        )

from dotenv import load_dotenv

from binance_sdk_spot.spot import (
    Spot,
    ConfigurationRestAPI,
    ConfigurationWebSocketAPI,
    ConfigurationWebSocketStreams,
)
from binance_sdk_spot.websocket_api.models import (
    OrderPlaceSideEnum,
    OrderPlaceTypeEnum,
)
from binance_common.constants import (
    SPOT_REST_API_PROD_URL,
    SPOT_WS_API_PROD_URL,
    SPOT_WS_STREAMS_PROD_URL,
    SPOT_REST_API_TESTNET_URL,
    SPOT_WS_API_TESTNET_URL,
    SPOT_WS_STREAMS_TESTNET_URL,
    SPOT_REST_API_DEMO_URL,
    SPOT_WS_API_DEMO_URL,
    SPOT_WS_STREAMS_DEMO_URL,
)

# ---------------------------------------------------------------------------
# Decimal precision
# ---------------------------------------------------------------------------
getcontext().prec = 28

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
logger = logging.getLogger("spot_bot")


# ===================================================================
# Configuration — everything from env
# ===================================================================

def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int = 0) -> int:
    v = _env(key)
    return int(v) if v else default


def _env_decimal(key: str, default: str = "0") -> Decimal:
    v = _env(key)
    return Decimal(v) if v else Decimal(default)


def _env_bool(key: str, default: bool = False) -> bool:
    v = _env(key).lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    return default


def _find_env_file() -> str:
    """Find env file: env or .env, in current dir or script dir."""
    for name in ("env", ".env"):
        for base in (".", os.path.dirname(os.path.abspath(__file__)) or "."):
            p = os.path.join(base, name)
            if os.path.isfile(p):
                return p
    return "env"


@dataclass
class BotConfig:
    """All bot parameters — loaded from environment variables."""
    # API credentials
    api_key: str = ""
    api_secret: str = ""

    # Mode: mainnet, testnet, demo
    mode: str = "demo"

    # Symbols (comma-separated)
    symbols: List[str] = field(default_factory=lambda: ["DOGEUSDT"])

    # Grid parameters
    grid_step_pct: Decimal = Decimal("0.2")       # step as % of price
    grid_levels: int = 10                          # levels per side (BUY + SELL)
    grid_order_size_usdt: Decimal = Decimal("5")   # base order size in USDT
    grid_min_order_usdt: Decimal = Decimal("1")    # minimum order size
    grid_max_order_usdt: Decimal = Decimal("50")   # maximum order size

    # Balance allocation
    balance_ratio: Decimal = Decimal("0.5")        # fraction of balance to use

    # Asymmetry (balance rebalancing through order sizes)
    asymmetry_enabled: bool = True
    asymmetry_low: Decimal = Decimal("0.3")        # USDT < 30% → shrink BUY size
    asymmetry_high: Decimal = Decimal("0.7")       # USDT > 70% → shrink SELL size

    # Inventory limit — prevents spending all capital in one direction
    # After N consecutive fills on one side without a fill on the other,
    # stop placing new orders on that side until opposite fill happens.
    # 0 = disabled (unlimited)
    inventory_limit: int = 0

    # Intervals (seconds)
    fill_poll_interval: float = 2.0     # REST fallback poll interval
    balance_poll_interval: float = 30.0  # how often to refresh balances
    ws_reconnect_delay: float = 5.0      # WS reconnect delay

    # Logging
    log_level: str = "INFO"

    # Rate limiter: orders per second (Binance limit=10, use 8 for safety)
    order_rate_limit: float = 8.0

    # Heartbeat / health check interval (seconds)
    health_interval: int = 60

    def load(self) -> None:
        """Load all values from environment."""
        load_dotenv(dotenv_path=_find_env_file(), override=False)

        self.api_key = _env("BINANCE_API_KEY")
        self.api_secret = _env("BINANCE_API_SECRET")
        self.mode = _env("MODE", "demo").lower()
        if self.mode not in ("mainnet", "testnet", "demo"):
            logger.warning(f"Unknown MODE={self.mode}, falling back to demo")
            self.mode = "demo"

        raw = _env("SYMBOLS", "DOGEUSDT")
        self.symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]

        self.grid_step_pct = _env_decimal("GRID_STEP_PCT", "0.2")
        self.grid_levels = _env_int("GRID_LEVELS", 10)
        self.grid_order_size_usdt = _env_decimal("GRID_ORDER_SIZE_USDT", "5")
        self.grid_min_order_usdt = _env_decimal("GRID_MIN_ORDER_USDT", "1")
        self.grid_max_order_usdt = _env_decimal("GRID_MAX_ORDER_USDT", "50")

        self.balance_ratio = _env_decimal("BALANCE_RATIO", "0.5")

        self.asymmetry_enabled = _env_bool("ASYMMETRY_ENABLED", True)
        self.asymmetry_low = _env_decimal("ASYMMETRY_LOW", "0.3")
        self.asymmetry_high = _env_decimal("ASYMMETRY_HIGH", "0.7")

        self.inventory_limit = _env_int("INVENTORY_LIMIT", 0)

        self.fill_poll_interval = float(_env_decimal("FILL_POLL_INTERVAL", "2"))
        self.balance_poll_interval = float(_env_decimal("BALANCE_POLL_INTERVAL", "30"))
        self.ws_reconnect_delay = float(_env_decimal("WS_RECONNECT_DELAY", "5"))

        self.log_level = _env("LOG_LEVEL", "INFO")
        self.health_interval = _env_int("HEALTH_INTERVAL", 60)
        self.order_rate_limit = float(_env_decimal("ORDER_RATE_LIMIT", "8"))


# ===================================================================
# Symbol Info — from exchange
# ===================================================================

@dataclass
class SymbolInfo:
    """Trading filters for a symbol, retrieved from exchange info."""
    symbol: str = ""
    base_asset: str = ""
    quote_asset: str = ""
    status: str = "TRADING"

    # PRICE_FILTER
    tick_size: Decimal = Decimal("0.00000001")
    min_price: Decimal = Decimal("0")
    max_price: Decimal = Decimal("99999999")

    # LOT_SIZE
    step_size: Decimal = Decimal("0.00000001")
    min_qty: Decimal = Decimal("0")
    max_qty: Decimal = Decimal("99999999")

    # NOTIONAL / MIN_NOTIONAL
    min_notional: Decimal = Decimal("1")

    # MAX_NUM_ORDERS
    max_num_orders: int = 200

    # Computed precision
    price_precision: int = 8
    qty_precision: int = 8

    def round_price(self, price: Decimal) -> Decimal:
        """Round price down to tick size."""
        if self.tick_size == 0:
            return price
        return (price / self.tick_size).to_integral_value(rounding=ROUND_DOWN) * self.tick_size

    def round_price_up(self, price: Decimal) -> Decimal:
        """Round price up to tick size."""
        if self.tick_size == 0:
            return price
        return (price / self.tick_size).to_integral_value(rounding=ROUND_UP) * self.tick_size

    def round_qty(self, qty: Decimal) -> Decimal:
        """Round quantity down to step size."""
        if self.step_size == 0:
            return qty
        return (qty / self.step_size).to_integral_value(rounding=ROUND_DOWN) * self.step_size

    def round_qty_up(self, qty: Decimal) -> Decimal:
        """Round quantity up to step size."""
        if self.step_size == 0:
            return qty
        return (qty / self.step_size).to_integral_value(rounding=ROUND_UP) * self.step_size

    def format_price(self, price: Decimal) -> str:
        """Format price for API call (fixed-point, no scientific notation)."""
        return f"{price:.{self.price_precision}f}"

    def format_qty(self, qty: Decimal) -> str:
        """Format quantity for API call (fixed-point, no scientific notation)."""
        return f"{qty:.{self.qty_precision}f}"


def _unwrap(obj: Any) -> Any:
    """Unwrap WS API response: OneOf wrapper → actual_instance, then drill into .result.

    WS API responses come in three forms:
      1. OneOf Pydantic wrapper (has .actual_instance) — unwrap first
      2. Raw dict {"id": "...", "status": 200, "result": {...}, "rateLimits": [...]}
         — must drill into obj["result"] (dict has no .result attribute)
      3. Pydantic model with .result attribute — drill into obj.result
    """
    # 1. Handle OneOf wrapper (e.g., TickerPriceResponse has actual_instance)
    if hasattr(obj, "actual_instance") and obj.actual_instance is not None:
        obj = obj.actual_instance
    # 2. Handle dict with "result" key (raw WS API response)
    if isinstance(obj, dict) and "result" in obj and obj["result"] is not None:
        obj = obj["result"]
    # 3. Handle Pydantic model with .result attribute
    elif hasattr(obj, "result") and obj.result is not None:
        obj = obj.result
    return obj


def _get_attr(obj: Any, *names: str, default: Any = None) -> Any:
    """Try multiple attribute names (snake_case, camelCase, dict key)."""
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        else:
            if hasattr(obj, name):
                return getattr(obj, name)
    return default


def parse_symbol_info(raw: Any, symbol: str) -> SymbolInfo:
    """Parse exchange info response into SymbolInfo."""
    info = SymbolInfo(symbol=symbol)

    # Unwrap OneOf + .result wrapper
    raw = _unwrap(raw)

    # Navigate response structure
    sym = None
    if isinstance(raw, dict):
        syms = raw.get("symbols", [])
        for s in syms:
            if s.get("symbol") == symbol:
                sym = s
                break
    elif hasattr(raw, "symbols"):
        for s in raw.symbols:
            s_sym = getattr(s, "symbol", "")
            if s_sym == symbol:
                sym = s
                break

    if sym is None:
        logger.warning(f"[{symbol}] Not found in exchange info, using defaults")
        logger.debug(f"[{symbol}] Raw exchange info type={type(raw).__name__}, "
                      f"keys={list(raw.keys()) if isinstance(raw, dict) else 'N/A'}")
        if isinstance(raw, dict) and "symbols" in raw:
            logger.debug(f"[{symbol}] symbols count={len(raw['symbols'])}, "
                          f"first few={[s.get('symbol','?') if isinstance(s,dict) else getattr(s,'symbol','?') for s in raw['symbols'][:5]]}")
        # Fallback: определяем base/quote из названия символа
        if symbol.endswith("USDT"):
            info.base_asset = symbol[:-4]
            info.quote_asset = "USDT"
        elif symbol.endswith("BUSD"):
            info.base_asset = symbol[:-4]
            info.quote_asset = "BUSD"
        elif symbol.endswith("BTC"):
            info.base_asset = symbol[:-3]
            info.quote_asset = "BTC"
        elif symbol.endswith("ETH"):
            info.base_asset = symbol[:-3]
            info.quote_asset = "ETH"
        elif symbol.endswith("BNB"):
            info.base_asset = symbol[:-3]
            info.quote_asset = "BNB"
        else:
            info.base_asset = symbol[:-4] if len(symbol) > 4 else symbol
            info.quote_asset = "USDT"
        return info

    # Base/quote assets — try both snake_case and camelCase
    info.base_asset = _get_attr(sym, "base_asset", "baseAsset", default=symbol[:-4])
    info.quote_asset = _get_attr(sym, "quote_asset", "quoteAsset", default="USDT")
    info.status = _get_attr(sym, "status", default="TRADING")

    # Parse filters (OneOf wrappers → actual_instance)
    filters = _get_attr(sym, "filters", default=[])
    for f in filters:
        f = _unwrap(f)  # Unwrap OneOf
        ftype = _get_attr(f, "filter_type", "filterType", default="")

        if ftype == "PRICE_FILTER":
            info.tick_size = _safe_decimal(_get_attr(f, "tick_size", "tickSize", default="0.01"))
            info.min_price = _safe_decimal(_get_attr(f, "min_price", "minPrice", default="0"))
            info.max_price = _safe_decimal(_get_attr(f, "max_price", "maxPrice", default="99999999"))
        elif ftype == "LOT_SIZE":
            info.step_size = _safe_decimal(_get_attr(f, "step_size", "stepSize", default="0.001"))
            info.min_qty = _safe_decimal(_get_attr(f, "min_qty", "minQty", default="0"))
            info.max_qty = _safe_decimal(_get_attr(f, "max_qty", "maxQty", default="99999999"))
        elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
            info.min_notional = _safe_decimal(_get_attr(f, "min_notional", "minNotional", default="1"))
        elif ftype == "MAX_NUM_ORDERS":
            info.max_num_orders = _safe_int(_get_attr(f, "max_num_orders", "maxNumOrders", default="200"))

    # Compute precision from step sizes
    info.price_precision = _decimal_places(info.tick_size)
    info.qty_precision = _decimal_places(info.step_size)

    return info


def _safe_decimal(v: str) -> Decimal:
    try:
        return Decimal(str(v))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def _safe_int(v: str) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


def _decimal_places(d: Decimal) -> int:
    """Count decimal places in a Decimal (e.g., 0.001 → 3)."""
    if d == 0:
        return 8
    normalized = d.normalize()
    sign, digits, exponent = normalized.as_tuple()
    if exponent >= 0:
        return 0
    return -exponent


# ===================================================================
# Grid Level
# ===================================================================

# ===================================================================
# Bot order prefix for clientOrderId
# ===================================================================

BOT_ORDER_PREFIX = "spotbot_"


def make_client_order_id(symbol: str, side: str, index: int) -> str:
    """Generate a clientOrderId for bot orders.

    Format: spotbot_{symbol}_{side}_{index}
    Example: spotbot_DOGEUSDT_BUY_3

    Binance rules: max 36 chars, alphanumeric + underscore + hyphen.
    """
    return f"{BOT_ORDER_PREFIX}{symbol}_{side}_{index}"


def is_bot_order(client_order_id: Optional[str]) -> bool:
    """Check if a clientOrderId belongs to our bot."""
    return client_order_id is not None and client_order_id.startswith(BOT_ORDER_PREFIX)


@dataclass
class GridLevel:
    """A single grid level with order tracking."""
    price: Decimal
    quantity: Decimal
    side: str                        # "BUY" or "SELL"
    order_id: Optional[int] = None
    client_order_id: Optional[str] = None
    status: str = "NEW"             # NEW, PLACED, FILLED, CANCELED, ERROR


# ===================================================================
# Symbol Grid Manager
# ===================================================================

class SymbolGrid:
    """Manages BUY and SELL grids for one symbol."""

    def __init__(self, symbol: str, info: SymbolInfo, config: BotConfig):
        self.symbol = symbol
        self.info = info
        self.config = config

        # Price tracking
        self.current_price: Decimal = Decimal("0")   # обновляется bookTicker (mid-price)
        self.center_price: Decimal = Decimal("0")    # anchor для построения сетки

        # Grids
        self.buy_levels: List[GridLevel] = []
        self.sell_levels: List[GridLevel] = []

        # Balance tracking (for this symbol only)
        self.coin_free: Decimal = Decimal("0")
        self.coin_in_sell_orders: Decimal = Decimal("0")
        self.usdt_free: Decimal = Decimal("0")
        self.usdt_in_buy_orders: Decimal = Decimal("0")

        # Set of order_ids we know are placed (for fill detection)
        self.placed_order_ids: Set[int] = set()

        # Level index counter for clientOrderId generation
        self._level_index: int = 0

        # Statistics
        self.fills_buy: int = 0
        self.fills_sell: int = 0
        self.cycles: int = 0              # completed buy→sell or sell→buy cycles
        self.realized_pnl: Decimal = Decimal("0")
        self.commission_paid_usdt: Decimal = Decimal("0")  # в USDT-эквиваленте

        # Средневзвешенная цена покупки (для PnL расчёта)
        self._avg_buy_price: Decimal = Decimal("0")
        self._avg_buy_coin: Decimal = Decimal("0")   # кол-во монет в «открытой» позиции

        # Инвентарный лимит: счётчик последовательных fills в одну сторону
        # BUY fill → _consec_buy += 1, _consec_sell = 0
        # SELL fill → _consec_sell += 1, _consec_buy = 0
        # Если _consec_buy >= inventory_limit → не ставим BUY
        # Если _consec_sell >= inventory_limit → не ставим SELL
        self._consec_buy: int = 0
        self._consec_sell: int = 0

        # Сторона, на которой стоит пауза инвентарного лимита
        # None = нет паузы, "BUY" = не ставим BUY, "SELL" = не ставим SELL
        self._inventory_pause: Optional[str] = None

        # Anchor цена для grid rebuild — заполняется при fill, используется при shift.
        # Ключ = сторона, которую нужно сдвинуть ("SELL" или "BUY"),
        # значение = глубочайшая fill_price для этой стороны.
        # BUY fill  → shift SELL → anchor = самая НИЗКАЯ BUY fill price
        # SELL fill → shift BUY  → anchor = самая ВЫСОКАЯ SELL fill price
        # Это гарантирует что противоположная сетка строится от реальной
        # позиции, а не от устаревшего center_price — gap не растёт.
        self._shift_anchor: Dict[str, Decimal] = {}

    # ----- Properties -----

    @property
    def base_asset(self) -> str:
        return self.info.base_asset

    def calc_step(self, anchor: Decimal) -> Decimal:
        """Step in price units — всегда считается от anchor (center/fill) цены,
        не от volatile current_price (bookTicker)."""
        if anchor <= 0:
            return Decimal("0")
        return anchor * self.config.grid_step_pct / Decimal("100")

    def total_coin(self) -> Decimal:
        return self.coin_free + self.coin_in_sell_orders

    def total_usdt(self) -> Decimal:
        return self.usdt_free + self.usdt_in_buy_orders

    def total_value_usdt(self) -> Decimal:
        return self.total_coin() * self.current_price + self.total_usdt()

    def usdt_ratio(self) -> Decimal:
        """Ratio of USDT to total value (0.0 - 1.0)."""
        total = self.total_value_usdt()
        if total <= 0:
            return Decimal("0.5")
        return self.total_usdt() / total

    # ----- Order size with asymmetry -----

    def calc_order_size_usdt(self, side: str) -> Decimal:
        """Calculate order size in USDT, applying asymmetry if enabled."""
        size = self.config.grid_order_size_usdt

        if self.config.asymmetry_enabled:
            ratio = self.usdt_ratio()

            if side == "BUY" and ratio < self.config.asymmetry_low:
                # Little USDT left → reduce BUY size proportionally
                factor = ratio / self.config.asymmetry_low if self.config.asymmetry_low > 0 else Decimal("0")
                size = size * factor
            elif side == "SELL" and ratio > self.config.asymmetry_high:
                # Little coin left → reduce SELL size proportionally
                coin_ratio = Decimal("1") - ratio
                threshold = Decimal("1") - self.config.asymmetry_high
                factor = coin_ratio / threshold if threshold > 0 else Decimal("0")
                size = size * factor

        # Clamp
        size = max(size, self.config.grid_min_order_usdt)
        size = min(size, self.config.grid_max_order_usdt)
        return size

    # ----- Grid building -----

    def build_grids(self) -> None:
        """Build BUY and SELL grid levels from current price.
        center_price = current_price на момент начальной постройки."""
        step = self.calc_step(self.current_price)
        if step <= 0:
            logger.warning(f"[{self.symbol}] Cannot build grids: step={step}, price={self.current_price}")
            return

        self.center_price = self.current_price
        self.buy_levels = []
        self.sell_levels = []

        for i in range(1, self.config.grid_levels + 1):
            # BUY level: below center price
            #buy_price = self.info.round_price(self.center_price - step * i)
            buy_price = self.info.round_price(self.center_price - step * (Decimal(i) - Decimal("0.5")))
            if buy_price > 0 and buy_price >= self.info.min_price:
                buy_size = self.calc_order_size_usdt("BUY")
                buy_qty = self.info.round_qty(buy_size / buy_price) if buy_price > 0 else Decimal("0")
                # Bump qty to meet min_notional if needed
                if buy_qty > 0 and (buy_qty * buy_price) < self.info.min_notional:
                    buy_qty = self.info.round_qty_up(self.info.min_notional / buy_price)
                if buy_qty >= self.info.min_qty and (buy_qty * buy_price) >= self.info.min_notional:
                    self.buy_levels.append(GridLevel(price=buy_price, quantity=buy_qty, side="BUY"))

            # SELL level: above center price
            #sell_price = self.info.round_price_up(self.center_price + step * i)
            sell_price = self.info.round_price_up(self.center_price + step * (Decimal(i) - Decimal("0.5")))
            if sell_price <= self.info.max_price:
                sell_size = self.calc_order_size_usdt("SELL")
                sell_qty = self.info.round_qty(sell_size / sell_price) if sell_price > 0 else Decimal("0")
                # Bump qty to meet min_notional if needed
                if sell_qty > 0 and (sell_qty * sell_price) < self.info.min_notional:
                    sell_qty = self.info.round_qty_up(self.info.min_notional / sell_price)
                if sell_qty >= self.info.min_qty and (sell_qty * sell_price) >= self.info.min_notional:
                    self.sell_levels.append(GridLevel(price=sell_price, quantity=sell_qty, side="SELL"))

        logger.info(
            f"[{self.symbol}] Grids built: {len(self.buy_levels)} BUY, "
            f"{len(self.sell_levels)} SELL, step={step:.8f}, "
            f"center={self.center_price}"
        )

    def rebuild_sell_grid(self, anchor_price: Decimal) -> None:
        """Rebuild SELL grid from anchor_price (после BUY fill → shift SELL).

        anchor_price = fill_price BUY ордера.
        SELL сетка строится ВЫШЕ anchor_price:
          SELL_1 = anchor + 1*step, SELL_2 = anchor + 2*step, ...
        Это гарантирует что SELL сетка «закрывает» BUY позицию с шагом прибыли.
        """
        step = self.calc_step(anchor_price)
        if step <= 0:
            return

        self.center_price = anchor_price
        new_levels = []
        for i in range(1, self.config.grid_levels + 1):
            sell_price = self.info.round_price_up(anchor_price + step * i)
            if sell_price <= self.info.max_price:
                sell_size = self.calc_order_size_usdt("SELL")
                sell_qty = self.info.round_qty(sell_size / sell_price) if sell_price > 0 else Decimal("0")
                if sell_qty > 0 and (sell_qty * sell_price) < self.info.min_notional:
                    sell_qty = self.info.round_qty_up(self.info.min_notional / sell_price)
                if sell_qty >= self.info.min_qty and (sell_qty * sell_price) >= self.info.min_notional:
                    new_levels.append(GridLevel(price=sell_price, quantity=sell_qty, side="SELL"))

        self.sell_levels = new_levels
        logger.debug(f"[{self.symbol}] SELL grid rebuilt: {len(new_levels)} levels from anchor={anchor_price}")

    def rebuild_buy_grid(self, anchor_price: Decimal) -> None:
        """Rebuild BUY grid from anchor_price (после SELL fill → shift BUY).

        anchor_price = fill_price SELL ордера.
        BUY сетка строится НИЖЕ anchor_price:
          BUY_1 = anchor - 1*step, BUY_2 = anchor - 2*step, ...
        Это гарантирует что BUY сетка «закрывает» SELL позицию с шагом прибыли.
        """
        step = self.calc_step(anchor_price)
        if step <= 0:
            return

        self.center_price = anchor_price
        new_levels = []
        for i in range(1, self.config.grid_levels + 1):
            buy_price = self.info.round_price(anchor_price - step * i)
            if buy_price > 0 and buy_price >= self.info.min_price:
                buy_size = self.calc_order_size_usdt("BUY")
                buy_qty = self.info.round_qty(buy_size / buy_price) if buy_price > 0 else Decimal("0")
                if buy_qty > 0 and (buy_qty * buy_price) < self.info.min_notional:
                    buy_qty = self.info.round_qty_up(self.info.min_notional / buy_price)
                if buy_qty >= self.info.min_qty and (buy_qty * buy_price) >= self.info.min_notional:
                    new_levels.append(GridLevel(price=buy_price, quantity=buy_qty, side="BUY"))

        self.buy_levels = new_levels
        logger.debug(f"[{self.symbol}] BUY grid rebuilt: {len(new_levels)} levels from anchor={anchor_price}")

    # ----- PnL tracking -----

    def update_avg_buy(self, fill_price: Decimal, fill_qty: Decimal) -> None:
        """Обновить средневзвешенную цену покупки при BUY fill."""
        total_coin = self._avg_buy_coin + fill_qty
        if total_coin > 0:
            self._avg_buy_price = (
                (self._avg_buy_price * self._avg_buy_coin + fill_price * fill_qty)
                / total_coin
            )
        self._avg_buy_coin = total_coin

    def calc_realized_pnl(self, sell_price: Decimal, sell_qty: Decimal) -> Decimal:
        """Рассчитать реализованный PnL при SELL fill.

        PnL = (sell_price - avg_buy_price) × sell_qty
        Учитываем что мы продаём монеты, купленные по средней цене.
        """
        if self._avg_buy_coin <= 0 or self._avg_buy_price <= 0:
            return Decimal("0")

        # Продаём не больше чем есть в «открытой» позиции
        qty = min(sell_qty, self._avg_buy_coin)
        pnl = (sell_price - self._avg_buy_price) * qty

        # Уменьшаем «открытую» позицию
        self._avg_buy_coin -= qty
        if self._avg_buy_coin <= 0:
            self._avg_buy_coin = Decimal("0")
            self._avg_buy_price = Decimal("0")

        return pnl

    # ----- Status -----

    def status_line(self) -> str:
        """One-line status for logging."""
        coin_val = self.total_coin() * self.current_price
        total = self.total_value_usdt()
        return (
            f"[{self.symbol}] price={self.current_price} center={self.center_price} "
            f"coin={self.total_coin():.4f} ({coin_val:.2f} USDT) "
            f"usdt={self.total_usdt():.2f} total={total:.2f} "
            f"ratio={self.usdt_ratio():.2f} "
            f"fills: {self.fills_buy}B/{self.fills_sell}S "
            f"inv_pause={self._inventory_pause or '-'} "
            f"pnl={self.realized_pnl:.4f} comm={self.commission_paid_usdt:.4f} USDT"
        )


# ===================================================================
# Main Bot
# ===================================================================

class SpotBot:
    """Multi-symbol spot market maker bot — fully WebSocket-based."""

    def __init__(self):
        self.config = BotConfig()
        self.config.load()

        logging.getLogger("spot_bot").setLevel(
            getattr(logging, self.config.log_level, logging.INFO)
        )

        self.client: Optional[Spot] = None
        self.grids: Dict[str, SymbolGrid] = {}

        # WebSocket API connection (trading + account + user data)
        self.ws_api_conn = None
        # WebSocket Streams connection (market data)
        self.ws_streams_conn = None

        # User Data Stream subscription
        self._uds_stream = None

        # Commission tracking
        self.maker_commission: Decimal = Decimal("0.001")   # 0.1% default
        self.taker_commission: Decimal = Decimal("0.001")
        self.bnb_discount: Decimal = Decimal("0.75")        # множитель: 0.75 = платишь 75% ставки
        self.bnb_discount_enabled: bool = False
        self.effective_commission: Decimal = Decimal("0.00075")  # with BNB: 0.1% * 0.75

        # Runtime state
        self._running = False
        self._last_health = 0.0

        # Fill queue — orders we need to process fills for
        self._fill_events: asyncio.Queue = None

        # Shift lock per symbol — prevents concurrent grid shifts
        self._shift_locks: Dict[str, asyncio.Lock] = {}

        # Set of already-processed fill order IDs — prevents UDS/poll double processing
        self._processed_fill_ids: Set[int] = set()

        # UDS resubscribe flag — set by EventStreamTerminated, cleared by main loop
        self._uds_needs_resubscribe: bool = False

        # WS health monitoring
        self._ws_last_ok: float = 0.0  # timestamp of last successful WS operation
        self._ws_check_interval: float = 30.0  # seconds between health checks

        # Rate limiter for order operations (prevents Binance ban)
        self._order_limiter: Optional[OrderRateLimiter] = None

    # ----- URL helpers -----

    @property
    def rest_url(self) -> str:
        urls = {
            "mainnet": SPOT_REST_API_PROD_URL,
            "testnet": SPOT_REST_API_TESTNET_URL,
            "demo": SPOT_REST_API_DEMO_URL,
        }
        return urls.get(self.config.mode, SPOT_REST_API_DEMO_URL)

    @property
    def ws_api_url(self) -> str:
        urls = {
            "mainnet": SPOT_WS_API_PROD_URL,
            "testnet": SPOT_WS_API_TESTNET_URL,
            "demo": SPOT_WS_API_DEMO_URL,
        }
        return urls.get(self.config.mode, SPOT_WS_API_DEMO_URL)

    @property
    def ws_streams_url(self) -> str:
        urls = {
            "mainnet": SPOT_WS_STREAMS_PROD_URL,
            "testnet": SPOT_WS_STREAMS_TESTNET_URL,
            "demo": SPOT_WS_STREAMS_DEMO_URL,
        }
        return urls.get(self.config.mode, SPOT_WS_STREAMS_DEMO_URL)

    # ----- Initialization -----

    def init_client(self) -> None:
        """Create the Spot client with REST + WS API + WS Streams."""
        cfg_rest = ConfigurationRestAPI(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            base_path=self.rest_url,
            timeout=10000,
            retries=3,
        )
        cfg_ws_api = ConfigurationWebSocketAPI(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
            stream_url=self.ws_api_url,
            reconnect_delay=int(self.config.ws_reconnect_delay * 1000),
        )
        cfg_ws_streams = ConfigurationWebSocketStreams(
            stream_url=self.ws_streams_url,
            reconnect_delay=5000,
        )

        self.client = Spot(
            config_rest_api=cfg_rest,
            config_ws_api=cfg_ws_api,
            config_ws_streams=cfg_ws_streams,
        )
        mode = self.config.mode.upper()
        logger.info(f"Client initialized [{mode}]")
        logger.info(f"  REST:       {self.rest_url}")
        logger.info(f"  WS API:     {self.ws_api_url}")
        logger.info(f"  WS Streams: {self.ws_streams_url}")

    # ----- WS API: Exchange info -----

    async def fetch_exchange_info(self) -> Dict[str, SymbolInfo]:
        """Get symbol filters from exchange info via WS API.

        Строго по Binance WS API:
          - 1 символ: параметр symbol (string) — один запрос
          - N символов: параметр symbols (JSON array string) — '["BTCUSDT","DOGEUSDT"]'
        SDK не выбрасывает исключения при ошибках WS API — проверяем код ответа.
        """
        result = {}
        if self.ws_api_conn is None:
            return await self._fetch_exchange_info_rest()

        try:
            # Batch запрос — все символы в одном вызове
            # Binance WS API требует JSON-массив: symbols='["DOGEUSDT"]'
            # НЕ plain string или comma-separated!
            if len(self.config.symbols) == 1:
                # Одиночный символ — используем параметр symbol (не symbols)
                resp = await self.ws_api_conn.exchange_info(symbol=self.config.symbols[0])
            else:
                # Несколько символов — параметр symbols как JSON-массив
                symbols_param = json.dumps(self.config.symbols)
                resp = await self.ws_api_conn.exchange_info(symbols=symbols_param)
            raw = resp.data()
            logger.debug(f"exchange_info raw type={type(raw).__name__}")

            # Проверяем ошибку WS API (SDK не выбрасывает исключения!)
            unwrapped = _unwrap(raw)
            if isinstance(unwrapped, dict) and "code" in unwrapped and unwrapped.get("code", 200) < 0:
                raise ValueError(f"WS API error: code={unwrapped['code']} msg={unwrapped.get('msg', '?')}")

            # Parse для каждого символа из единого ответа
            for symbol in self.config.symbols:
                info = parse_symbol_info(raw, symbol)
                result[symbol] = info
                logger.info(
                    f"[{symbol}] Filters: tickSize={info.tick_size}, "
                    f"stepSize={info.step_size}, minQty={info.min_qty}, "
                    f"minNotional={info.min_notional}, "
                    f"pricePrec={info.price_precision}, qtyPrec={info.qty_precision}"
                )
        except Exception as e:
            logger.warning(f"WS API batch exchange_info failed: {e}")
            # Fallback: по одному через параметр symbol (не symbols)
            try:
                for symbol in self.config.symbols:
                    resp = await self.ws_api_conn.exchange_info(symbol=symbol)
                    raw = resp.data()

                    # Проверяем ошибку WS API
                    unwrapped = _unwrap(raw)
                    if isinstance(unwrapped, dict) and "code" in unwrapped and unwrapped.get("code", 200) < 0:
                        raise ValueError(f"WS API error for {symbol}: code={unwrapped['code']} msg={unwrapped.get('msg', '?')}")

                    info = parse_symbol_info(raw, symbol)
                    result[symbol] = info
                    logger.info(
                        f"[{symbol}] Filters: tickSize={info.tick_size}, "
                        f"stepSize={info.step_size}, minQty={info.min_qty}, "
                        f"minNotional={info.min_notional}, "
                        f"pricePrec={info.price_precision}, qtyPrec={info.qty_precision}"
                    )
            except Exception as e2:
                logger.warning(f"WS API exchange_info per-symbol also failed: {e2}, falling back to REST")
                return await self._fetch_exchange_info_rest()
        return result

    async def _fetch_exchange_info_rest(self) -> Dict[str, SymbolInfo]:
        """Fallback: get exchange info via REST."""
        result = {}
        for symbol in self.config.symbols:
            try:
                resp = self.client.rest_api.exchange_info(symbol=symbol)
                raw = resp.data()
                info = parse_symbol_info(raw, symbol)
                result[symbol] = info
                logger.info(
                    f"[{symbol}] Filters (REST): tickSize={info.tick_size}, "
                    f"stepSize={info.step_size}, minQty={info.min_qty}, "
                    f"minNotional={info.min_notional}"
                )
            except Exception as e:
                logger.error(f"[{symbol}] Failed to get exchange info: {e}")
        return result

    # ----- WS API: Commission -----

    async def fetch_commission(self, symbol: str) -> None:
        """Get commission rates for a symbol via WS API."""
        if self.ws_api_conn is None:
            return await self._fetch_commission_rest(symbol)

        try:
            resp = await self.ws_api_conn.account_commission(symbol=symbol)
            data = _unwrap(resp.data())
            self._parse_commission(data)
        except Exception as e:
            logger.warning(f"WS API commission failed: {e}, falling back to REST")
            await self._fetch_commission_rest(symbol)

    async def _fetch_commission_rest(self, symbol: str) -> None:
        """Fallback: get commission via REST."""
        try:
            resp = self.client.rest_api.account_commission(symbol=symbol)
            data = resp.data()
            self._parse_commission(data)
        except Exception as e:
            logger.warning(f"Could not fetch commission: {e}")

    def _parse_commission(self, data: Any) -> None:
        """Parse commission data (shared by WS API and REST)."""
        if hasattr(data, 'result') and data.result is not None:
            data = data.result
        elif isinstance(data, dict) and 'result' in data:
            data = data['result']
        logger.debug(f"Commission data type={type(data).__name__}")

        if isinstance(data, dict):
            std = data.get("standardCommission", {})
            self.maker_commission = Decimal(str(std.get("maker", "0.001")))
            self.taker_commission = Decimal(str(std.get("taker", "0.001")))
            disc = data.get("discount", {})
            self.bnb_discount_enabled = disc.get("enabledForAccount", False)
            self.bnb_discount = Decimal(str(disc.get("discount", "0.75")))
        elif hasattr(data, "standard_commission"):
            self.maker_commission = Decimal(str(data.standard_commission.maker))
            self.taker_commission = Decimal(str(data.standard_commission.taker))
            self.bnb_discount_enabled = getattr(data.discount, "enabled_for_account", False)
            self.bnb_discount = Decimal(str(getattr(data.discount, "discount", "0.75")))
        elif hasattr(data, "commission_rates"):
            cr = data.commission_rates
            self.maker_commission = Decimal(str(getattr(cr, "maker", "0.001")))
            self.taker_commission = Decimal(str(getattr(cr, "taker", "0.001")))
            self.bnb_discount_enabled = False
            self.bnb_discount = Decimal("0.75")  # множитель: 0.75 = платишь 75% ставки

        # BNB discount: 0.75 = платишь 75% от стандартной ставки (скидка 25%)
        # НЕ (1 - discount)! discount из API — это множитель, а не размер скидки.
        eff = self.taker_commission
        if self.bnb_discount_enabled:
            eff = eff * self.bnb_discount
        self.effective_commission = eff

        logger.info(
            f"Commission: maker={self.maker_commission} taker={self.taker_commission} "
            f"BNB_discount={'ON' if self.bnb_discount_enabled else 'OFF'} ({self.bnb_discount}) "
            f"effective={eff}"
        )

    # ----- WS API: Balances -----

    async def fetch_balances(self) -> Dict[str, tuple]:
        """Get account balances via WS API. Returns {asset: (free, locked)}."""
        if self.ws_api_conn is None:
            return await self._fetch_balances_rest()

        try:
            resp = await self.ws_api_conn.account_status(omit_zero_balances=True)
            raw = resp.data()
            logger.debug(f"account_status raw type={type(raw).__name__}")
            if hasattr(raw, 'result'):
                logger.debug(f"account_status .result type={type(raw.result).__name__}")
            if isinstance(raw, dict):
                logger.debug(f"account_status dict keys={list(raw.keys())}")
            data = _unwrap(raw)
            return self._parse_balances(data)
        except Exception as e:
            logger.warning(f"WS API account_status failed: {e}, falling back to REST")
            return await self._fetch_balances_rest()

    async def _fetch_balances_rest(self) -> Dict[str, tuple]:
        """Fallback: get balances via REST."""
        try:
            resp = self.client.rest_api.get_account()
            data = _unwrap(resp.data())
            return self._parse_balances(data)
        except Exception as e:
            logger.warning(f"Cannot fetch balances: {e}")
            return {}

    def _parse_balances(self, data: Any) -> Dict[str, tuple]:
        """Parse balance data (shared by WS API and REST).

        Возвращает словарь: asset → (free, locked) как tuple.
        Важно разделять free и locked — locked уже в ордерах,
        наш coin_free/usdt_free должен использовать только free.
        """
        if hasattr(data, 'result') and data.result is not None:
            data = data.result
        elif isinstance(data, dict) and 'result' in data:
            data = data['result']
        logger.debug(f"Balances data type={type(data).__name__}")

        # Возвращаем структурированные балансы: {asset: (free, locked)}
        balances: Dict[str, tuple] = {}
        bals = _get_attr(data, "balances", default=[])
        for b in bals:
            b = _unwrap(b)
            asset = _get_attr(b, "asset", default="")
            free = Decimal(str(_get_attr(b, "free", default="0")))
            locked = Decimal(str(_get_attr(b, "locked", default="0")))
            total = free + locked
            if total > 0:
                balances[asset] = (free, locked)

        usdt_free, usdt_locked = balances.get("USDT", (Decimal("0"), Decimal("0")))
        bnb_free, bnb_locked = balances.get("BNB", (Decimal("0"), Decimal("0")))
        usdt_total = usdt_free + usdt_locked
        bnb_total = bnb_free + bnb_locked
        logger.info(f"Balance: USDT={usdt_total:.2f} (free={usdt_free:.2f}), "
                    f"BNB={bnb_total:.4f} (free={bnb_free:.4f}), assets={len(balances)}")
        return balances

    # ----- WS API: Price -----

    async def fetch_price(self, symbol: str) -> Decimal:
        """Get current price for a symbol via WS API."""
        if self.ws_api_conn is None:
            return await self._fetch_price_rest(symbol)

        try:
            resp = await self.ws_api_conn.ticker_price(symbol=symbol)
            raw = resp.data()
            logger.debug(f"[{symbol}] ticker_price raw type={type(raw).__name__}")
            if hasattr(raw, 'actual_instance'):
                logger.debug(f"[{symbol}] ticker_price .actual_instance type={type(raw.actual_instance).__name__}")
            if isinstance(raw, dict):
                logger.debug(f"[{symbol}] ticker_price dict keys={list(raw.keys())}")
            data = _unwrap(raw)
            logger.debug(f"[{symbol}] ticker_price unwrapped type={type(data).__name__}")
            price_str = str(_get_attr(data, "price", default="0"))
            logger.debug(f"[{symbol}] ticker_price price_str={price_str}")
            return Decimal(price_str)
        except Exception as e:
            logger.warning(f"WS API ticker_price failed: {e}, falling back to REST")
            return await self._fetch_price_rest(symbol)

    async def _fetch_price_rest(self, symbol: str) -> Decimal:
        """Fallback: get price via REST."""
        try:
            resp = self.client.rest_api.ticker_price(symbol=symbol)
            data = _unwrap(resp.data())
            price_str = str(_get_attr(data, "price", default="0"))
            return Decimal(price_str)
        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch price: {e}")
            return Decimal("0")

    # ----- Setup -----

    async def setup_symbols(self) -> None:
        """Set up all configured symbols: info, price, grids.

        ПРИНЦИП: биржа — эталон. Сначала пробуем восстановить состояние
        из ордеров на бирже (recover_grids_from_exchange). Если ордеров
        нет (чистый старт) — строим новые сетки (build_grids).
        """
        symbol_infos = await self.fetch_exchange_info()
        balances = await self.fetch_balances()

        if self.config.symbols:
            await self.fetch_commission(self.config.symbols[0])

        n_symbols = max(1, len(self.config.symbols))

        for symbol in self.config.symbols:
            if symbol not in symbol_infos:
                logger.error(f"[{symbol}] No exchange info — skipping")
                continue

            info = symbol_infos[symbol]
            if info.status != "TRADING":
                logger.warning(f"[{symbol}] Status={info.status} — skipping")
                continue

            price = await self.fetch_price(symbol)
            if price <= 0:
                logger.error(f"[{symbol}] Invalid price — skipping")
                continue

            grid = SymbolGrid(symbol=symbol, info=info, config=self.config)
            grid.current_price = price

            # Балансы: balances теперь содержит {asset: (free, locked)}
            usdt_free, usdt_locked = balances.get("USDT", (Decimal("0"), Decimal("0")))
            coin_free, coin_locked = balances.get(info.base_asset, (Decimal("0"), Decimal("0")))

            grid.usdt_free = usdt_free * self.config.balance_ratio / Decimal(str(n_symbols))
            grid.coin_free = coin_free * self.config.balance_ratio / Decimal(str(n_symbols))

            logger.info(
                f"[{symbol}] Balance: USDT free={usdt_free:.2f} locked={usdt_locked:.2f}, "
                f"{info.base_asset} free={coin_free:.4f} locked={coin_locked:.4f}"
            )

            # Пробуем восстановить состояние из ордеров на бирже
            recovered = await self.recover_grids_from_exchange(grid, balances, n_symbols)

            if not recovered:
                # Нет ордеров на бирже — чистый старт, строим новые сетки
                grid.build_grids()
                logger.info(f"[{symbol}] Fresh start: built new grids from price={price}")
            else:
                logger.info(f"[{symbol}] Recovered grids from exchange")

            self.grids[symbol] = grid
            logger.info(
                f"[{symbol}] Setup: price={price}, BUY={len(grid.buy_levels)}, "
                f"SELL={len(grid.sell_levels)}, "
                f"usdt_free={grid.usdt_free:.2f}, {info.base_asset}_free={grid.coin_free:.4f}"
            )

    async def recover_grids_from_exchange(
        self, grid: SymbolGrid, balances: Dict[str, tuple], n_symbols: int
    ) -> bool:
        """Восстановить grid состояние из ордеров на бирже.

        Строго по репозитарию binance-connector-python:
          - open_orders_status(symbol) → список открытых ордеров
          - Каждый ордер: symbol, orderId, clientOrderId, price, origQty, side, type, status

        Логика восстановления:
          1. Запросить открытые ордера для символа
          2. Разделить на наши (clientOrderId начинается с spotbot_) и чужие
          3. Наши ордера → восстановить как GridLevel с status=PLACED
          4. Чужие ордера → не трогать, но учесть в балансе (они замораживают средства)
          5. Определить center_price из распределения ордеров
          6. Рассчитать свободные балансы с учётом и наших, и чужих ордеров

        Возвращает True если удалось восстановить (есть ордера),
        False если ордеров нет (нужен чистый старт).
        """
        try:
            if self.ws_api_conn is not None:
                # Строго по примеру:
                # examples/websocket_api/Account/open_orders_status.py
                resp = await self.ws_api_conn.open_orders_status(symbol=grid.symbol)
                data = _unwrap(resp.data())
            else:
                resp = self.client.rest_api.get_open_orders(symbol=grid.symbol)
                data = _unwrap(resp.data())
        except Exception as e:
            logger.warning(f"[{grid.symbol}] Failed to fetch open orders: {e}")
            return False

        # Нормализуем ответ в список ордеров
        order_list = data
        if isinstance(data, dict):
            for key in ("result", "orders", "data"):
                val = data.get(key)
                if isinstance(val, list):
                    order_list = val
                    break
        if not isinstance(order_list, list):
            order_list = [order_list] if order_list else []

        if not order_list:
            logger.info(f"[{grid.symbol}] No open orders on exchange — fresh start needed")
            return False

        # Разделяем на наши и чужие ордера
        our_orders = []     # Наши — восстановим как grid levels
        foreign_orders = [] # Чужие — не трогаем, но учитываем баланс

        for order in order_list:
            order = _unwrap(order)

            oid = _get_attr(order, "order_id", "orderId", default=None)
            cid = _get_attr(order, "client_order_id", "clientOrderId", default="")
            side = _get_attr(order, "side", default="")
            price_str = _get_attr(order, "price", default="0")
            qty_str = _get_attr(order, "orig_qty", "origQty", default="0")
            status = _get_attr(order, "status", default="NEW")

            if not oid or not side:
                continue

            if is_bot_order(cid):
                our_orders.append({
                    "order_id": int(oid),
                    "client_order_id": str(cid),
                    "side": side,
                    "price": _safe_decimal(price_str),
                    "quantity": _safe_decimal(qty_str),
                    "status": status,
                })
            else:
                foreign_orders.append({
                    "order_id": int(oid),
                    "client_order_id": str(cid) if cid else None,
                    "side": side,
                    "price": _safe_decimal(price_str),
                    "quantity": _safe_decimal(qty_str),
                    "status": status,
                })

        if not our_orders:
            logger.info(
                f"[{grid.symbol}] Open orders exist but none are ours "
                f"({len(foreign_orders)} foreign orders) — fresh start needed"
            )
            # Чужие ордера НЕ трогаем — они замораживают баланс,
            # но мы учтём это в balance_ratio
            return False

        # Восстанавливаем наши ордера как grid levels
        buy_levels = []
        sell_levels = []
        for o in our_orders:
            level = GridLevel(
                price=o["price"],
                quantity=o["quantity"],
                side=o["side"],
                order_id=o["order_id"],
                client_order_id=o["client_order_id"],
                status="PLACED",
            )
            grid.placed_order_ids.add(o["order_id"])

            if o["side"] == "BUY":
                buy_levels.append(level)
            else:
                sell_levels.append(level)

        # Сортируем: BUY по убыванию цены (ближе к центру — первый),
        # SELL по возрастанию цены (ближе к центру — первый)
        buy_levels.sort(key=lambda l: l.price, reverse=True)
        sell_levels.sort(key=lambda l: l.price)

        grid.buy_levels = buy_levels
        grid.sell_levels = sell_levels

        # Определяем center_price из распределения ордеров:
        # center = граница между самым высоким BUY и самым низким SELL
        max_buy_price = max((l.price for l in buy_levels), default=Decimal("0"))
        min_sell_price = min((l.price for l in sell_levels), default=Decimal("0"))

        if max_buy_price > 0 and min_sell_price > 0:
            grid.center_price = (max_buy_price + min_sell_price) / Decimal("2")
        elif max_buy_price > 0:
            step = grid.calc_step(max_buy_price)
            grid.center_price = max_buy_price + step
        elif min_sell_price > 0:
            step = grid.calc_step(min_sell_price)
            grid.center_price = min_sell_price - step
        else:
            grid.center_price = grid.current_price

        # Рассчитываем locked балансы из НАШИХ ордеров
        grid.usdt_in_buy_orders = sum(l.price * l.quantity for l in buy_levels)
        grid.coin_in_sell_orders = sum(l.quantity for l in sell_levels)

        # Рассчитываем locked из ЧУЖИХ ордеров (они тоже замораживают баланс)
        foreign_usdt_locked = Decimal("0")
        foreign_coin_locked = Decimal("0")
        for o in foreign_orders:
            if o["side"] == "BUY":
                foreign_usdt_locked += o["price"] * o["quantity"]
            else:
                foreign_coin_locked += o["quantity"]

        # Свободный баланс = API free × ratio / n_symbols - наши locked
        # Чужие locked уже вычтены из API free (Binance их заморозил)
        usdt_free, usdt_locked = balances.get("USDT", (Decimal("0"), Decimal("0")))
        coin_free, coin_locked = balances.get(grid.base_asset, (Decimal("0"), Decimal("0")))

        usdt_per = usdt_free * self.config.balance_ratio / Decimal(str(n_symbols))
        coin_per = coin_free * self.config.balance_ratio / Decimal(str(n_symbols))

        grid.usdt_free = max(Decimal("0"), usdt_per - grid.usdt_in_buy_orders)
        grid.coin_free = max(Decimal("0"), coin_per - grid.coin_in_sell_orders)

        logger.info(
            f"[{grid.symbol}] Recovered: {len(buy_levels)} BUY, {len(sell_levels)} SELL "
            f"(center={grid.center_price}), "
            f"{len(foreign_orders)} foreign orders "
            f"(USDT locked={foreign_usdt_locked:.2f}, "
            f"{grid.base_asset} locked={foreign_coin_locked:.4f})"
        )

        return True

    # ----- WS API: Order operations -----

    async def place_order(self, grid: SymbolGrid, level: GridLevel) -> bool:
        """Place a single LIMIT order via WS API.

        Строго по репозитарию binance-connector-python:
          - order_place(symbol, side, type, timeIn_force, quantity, price)
          - side = OrderPlaceSideEnum["BUY"].value
          - type = OrderPlaceTypeEnum["LIMIT"].value
          - Response: WebsocketApiResponse[OrderPlaceResponse]
          - .data() → OrderPlaceResponse → .result → .order_id
        """
        # Rate limiter — ждём слот перед отправкой ордера
        if self._order_limiter:
            await self._order_limiter.acquire()

        if self.ws_api_conn is None:
            return await self._place_order_rest(grid, level)

        try:
            price_str = grid.info.format_price(level.price)
            qty_str = grid.info.format_qty(level.quantity)
            side_val = OrderPlaceSideEnum[level.side].value

            # Генерируем clientOrderId с префиксом бота
            # Это позволяет отличать наши ордера от ручных при восстановлении
            if not level.client_order_id:
                grid._level_index += 1
                level.client_order_id = make_client_order_id(
                    grid.symbol, level.side, grid._level_index
                )

            resp = await self.ws_api_conn.order_place(
                symbol=grid.symbol,
                side=side_val,
                type=OrderPlaceTypeEnum["LIMIT"].value,
                time_in_force="GTC",
                quantity=qty_str,
                price=price_str,
                new_client_order_id=level.client_order_id,
            )
            data = _unwrap(resp.data())

            # Проверяем на ошибку WS API (SDK не выбрасывает исключения для ошибок)
            if isinstance(data, dict) and "error" in data:
                logger.warning(f"[{grid.symbol}] Order rejected (WS error): {data['error']}")
                level.status = "ERROR"
                return False

            # Parse response
            oid = _get_attr(data, "order_id", "orderId", default=None)
            cid = _get_attr(data, "client_order_id", "clientOrderId", default=None)

            if oid:
                level.order_id = int(oid)
                level.client_order_id = str(cid) if cid else None
                level.status = "PLACED"
                grid.placed_order_ids.add(level.order_id)

                # Обновляем баланс locked
                fill_value = level.price * level.quantity
                if level.side == "BUY":
                    grid.usdt_free -= fill_value
                    grid.usdt_in_buy_orders += fill_value
                else:
                    grid.coin_free -= level.quantity
                    grid.coin_in_sell_orders += level.quantity

                logger.debug(f"[{grid.symbol}] {level.side} placed: {qty_str} @ {price_str} id={oid}")
                return True
            else:
                logger.warning(f"[{grid.symbol}] Order rejected: {data}")
                level.status = "ERROR"
                return False

        except Exception as e:
            logger.error(f"[{grid.symbol}] WS API place {level.side} error: {e}")
            return await self._place_order_rest(grid, level)

    async def _place_order_rest(self, grid: SymbolGrid, level: GridLevel) -> bool:
        """Fallback: place order via REST.

        Rate limiter уже отработал в place_order() — тут не дублируем.
        """
        try:
            price_str = grid.info.format_price(level.price)
            qty_str = grid.info.format_qty(level.quantity)

            # Генерируем clientOrderId с префиксом бота
            if not level.client_order_id:
                grid._level_index += 1
                level.client_order_id = make_client_order_id(
                    grid.symbol, level.side, grid._level_index
                )

            resp = self.client.rest_api.new_order(
                symbol=grid.symbol,
                side=level.side,
                type="LIMIT",
                time_in_force="GTC",
                quantity=qty_str,
                price=price_str,
                new_client_order_id=level.client_order_id,
            )
            data = _unwrap(resp.data())

            oid = _get_attr(data, "order_id", "orderId", default=None)
            cid = _get_attr(data, "client_order_id", "clientOrderId", default=None)

            if oid:
                level.order_id = int(oid)
                level.client_order_id = str(cid) if cid else None
                level.status = "PLACED"
                grid.placed_order_ids.add(level.order_id)

                # Обновляем баланс locked
                fill_value = level.price * level.quantity
                if level.side == "BUY":
                    grid.usdt_free -= fill_value
                    grid.usdt_in_buy_orders += fill_value
                else:
                    grid.coin_free -= level.quantity
                    grid.coin_in_sell_orders += level.quantity

                logger.debug(f"[{grid.symbol}] {level.side} placed (REST): {qty_str} @ {price_str} id={oid}")
                return True
            else:
                logger.warning(f"[{grid.symbol}] Order rejected (REST): {data}")
                level.status = "ERROR"
                return False

        except Exception as e:
            logger.error(f"[{grid.symbol}] REST place {level.side} error: {e}")
            level.status = "ERROR"
            return False

    async def cancel_order(self, symbol: str, order_id: int) -> bool:
        """Cancel a single order via WS API.

        Строго по репозитарию binance-connector-python:
          - order_cancel(symbol, order_id=int)
          - SDK НЕ выбрасывает исключения для WS API ошибок!
          - Возвращает WebsocketApiResponse, чей .data() содержит ошибку:
            {"error": "Error received from server: {'code': -2011, 'msg': 'Unknown order sent.'}"}
          - Мы проверяем и парсим ошибки из response data.

        -2011 (Unknown order) → success (ордер уже исполнен или отменён).
        """
        # Rate limiter — cancel тоже считается как order operation
        if self._order_limiter:
            await self._order_limiter.acquire()

        if self.ws_api_conn is None:
            return await self._cancel_order_rest(symbol, order_id)

        try:
            resp = await self.ws_api_conn.order_cancel(
                symbol=symbol,
                order_id=int(order_id),
            )
            data = resp.data()

            # SDK оборачивает ошибки в dict {"error": "..."}
            if isinstance(data, dict):
                err = data.get("error")
                if err is not None:
                    return self._handle_cancel_error(symbol, order_id, err)

                # Проверяем код ошибки на верхнем уровне
                if "code" in data and data.get("code") != 200:
                    return self._handle_cancel_error(symbol, order_id, data)

            # Успешная отмена
            return True

        except Exception as e:
            err_str = str(e)
            if "-2011" in err_str:
                logger.debug(f"[{symbol}] Cancel {order_id}: already filled/gone (-2011)")
                return True
            logger.warning(f"[{symbol}] WS API cancel {order_id} exception: {e}")
            return await self._cancel_order_rest(symbol, order_id)

    def _handle_cancel_error(self, symbol: str, order_id: int, err: Any) -> bool:
        """Handle cancel error — extract error code and decide if it's OK.

        SDK форматы ошибок:
          {"error": "Error received from server: {'code': -2011, 'msg': 'Unknown order sent.'}"}
          {"error": {"code": -2011, "msg": "Unknown order sent."}}
          {"code": -2011, "msg": "Unknown order sent."}
        """
        if isinstance(err, dict):
            err_code = err.get("code", 0)
            err_msg = str(err.get("msg", ""))
        else:
            err_str = str(err)
            # Парсим строку ошибки от SDK: "Error received from server: {'code': -2011, ...}"
            err_code = 0
            err_msg = err_str
            # Пробуем извлечь код из строки
            import re
            code_match = re.search(r"'code':\s*(-?\d+)", err_str)
            if code_match:
                err_code = int(code_match.group(1))

        if err_code == -2011 or "-2011" in err_msg:
            logger.debug(f"[{symbol}] Cancel {order_id}: already filled/gone (-2011)")
            return True
        else:
            logger.warning(f"[{symbol}] Cancel {order_id} error: code={err_code} msg={err_msg}")
            return False

    async def _cancel_order_rest(self, symbol: str, order_id: int) -> bool:
        """Fallback: cancel order via REST.

        Rate limiter уже отработал в cancel_order() — тут не дублируем.
        """
        try:
            self.client.rest_api.delete_order(symbol=symbol, order_id=str(order_id))
            return True
        except Exception as e:
            err_str = str(e)
            if "-2011" in err_str:
                logger.debug(f"[{symbol}] REST cancel {order_id}: already filled/gone (-2011)")
                return True
            logger.debug(f"[{symbol}] REST cancel {order_id} error: {e}")
            return False

    async def cancel_all_symbol_orders(self, symbol: str) -> None:
        """Cancel ALL open orders for a symbol via WS API.

        Один API вызов — cancelAll считается как 1 order operation.
        Но для безопасности тоже пропускаем через rate limiter.
        """
        if self._order_limiter:
            await self._order_limiter.acquire()

        if self.ws_api_conn is None:
            return await self._cancel_all_symbol_orders_rest(symbol)

        try:
            await self.ws_api_conn.open_orders_cancel_all(symbol=symbol)
            logger.info(f"[{symbol}] All orders canceled (WS API)")
        except Exception as e:
            logger.warning(f"WS API cancel all failed: {e}, falling back to REST")
            await self._cancel_all_symbol_orders_rest(symbol)

    async def _cancel_all_symbol_orders_rest(self, symbol: str) -> None:
        """Fallback: cancel all via REST."""
        try:
            self.client.rest_api.delete_open_orders(symbol=symbol)
            logger.info(f"[{symbol}] All orders canceled (REST)")
        except Exception as e:
            logger.error(f"[{symbol}] Cancel all error: {e}")

    async def cancel_side_orders(self, grid: SymbolGrid, side: str) -> None:
        """Cancel all orders on one side (BUY or SELL) — CONCURRENT.

        -2011 (Unknown order) → success — ордер уже исполнен.
        FILLED ордера не отменяются.
        Параллельные отмены через asyncio.gather.
        Обновляет баланс locked при отмене.
        """
        levels = grid.buy_levels if side == "BUY" else grid.sell_levels

        to_cancel = []
        for level in levels:
            if level.order_id and level.status == "PLACED":
                grid.placed_order_ids.discard(level.order_id)
                to_cancel.append(level)
            elif level.order_id and level.status == "FILLED":
                grid.placed_order_ids.discard(level.order_id)

        if to_cancel:
            async def _cancel_one(lvl: GridLevel) -> bool:
                return await self.cancel_order(grid.symbol, lvl.order_id)

            results = await asyncio.gather(
                *[_cancel_one(lvl) for lvl in to_cancel],
                return_exceptions=True
            )
            for lvl, result in zip(to_cancel, results):
                if isinstance(result, Exception):
                    err_str = str(result)
                    if "-2011" in err_str:
                        lvl.status = "CANCELED"
                    else:
                        logger.warning(
                            f"[{grid.symbol}] Failed to cancel {side} id={lvl.order_id}: {result}"
                        )
                elif result:
                    lvl.status = "CANCELED"
                    # Возвращаем locked в free
                    fill_value = lvl.price * lvl.quantity
                    if side == "BUY":
                        grid.usdt_in_buy_orders -= fill_value
                        grid.usdt_free += fill_value
                    else:
                        grid.coin_in_sell_orders -= lvl.quantity
                        grid.coin_free += lvl.quantity
                else:
                    logger.warning(
                        f"[{grid.symbol}] Failed to cancel {side} id={lvl.order_id} "
                        f"— order may still be live on exchange"
                    )

    async def place_side_orders(self, grid: SymbolGrid, side: str) -> None:
        """Place all orders on one side (BUY or SELL) — CONCURRENT.

        Баланс-проверка:
          - BUY: суммарная стоимость NEW ордеров не должна превышать usdt_free
          - SELL: суммарное количество NEW ордеров не должно превышать coin_free
        Важно: xxx_free уже НЕ включает DOGE/USDT, locked в PLACED ордерах
        (place_order вычитает при каждом размещении). Поэтому проверяем
        только NEW уровни — PLACED уже учтены в уменьшенном free.

        Инвентарный лимит: если _inventory_pause == side — не ставим.
        Параллельные размещения через asyncio.gather.
        """
        # Инвентарный лимит — проверяем до попытки размещения
        if self.config.inventory_limit > 0 and grid._inventory_pause == side:
            paused_levels = sum(1 for l in (grid.buy_levels if side == "BUY" else grid.sell_levels)
                               if l.status == "NEW")
            if paused_levels > 0:
                logger.info(
                    f"[{grid.symbol}] INVENTORY PAUSE: skipping {paused_levels} {side} orders "
                    f"(consecutive fills limit={self.config.inventory_limit})"
                )
            return

        levels = grid.buy_levels if side == "BUY" else grid.sell_levels

        to_place = []
        skipped_buy = 0
        skipped_sell = 0
        new_qty_accumulated = Decimal("0")       # накопитель для NEW qty (SELL)
        new_usdt_accumulated = Decimal("0")      # накопитель для NEW USDT (BUY)

        for level in levels:
            if level.status != "NEW":
                continue

            level_value = level.price * level.quantity  # USDT стоимость ордера

            if side == "BUY":
                # Проверяем: накопленные NEW USDT + этот ордер ≤ usdt_free
                if new_usdt_accumulated + level_value > grid.usdt_free:
                    skipped_buy += 1
                    level.status = "ERROR"
                    continue
                new_usdt_accumulated += level_value

            elif side == "SELL":
                # Проверяем: накопленные NEW qty + этот ордер ≤ coin_free
                if new_qty_accumulated + level.quantity > grid.coin_free:
                    skipped_sell += 1
                    level.status = "ERROR"
                    continue
                new_qty_accumulated += level.quantity

            to_place.append(level)

        if skipped_buy > 0:
            logger.info(f"[{grid.symbol}] BUY skipped: {skipped_buy} levels "
                        f"(insufficient USDT: need>{grid.usdt_free:.2f})")
        if skipped_sell > 0:
            asset_name = grid.base_asset or grid.symbol.replace("USDT", "").replace("BUSD", "")
            logger.info(f"[{grid.symbol}] SELL skipped: {skipped_sell} levels "
                        f"(insufficient {asset_name}: need>{grid.coin_free:.4f})")

        if to_place:
            results = await asyncio.gather(
                *[self.place_order(grid, level) for level in to_place],
                return_exceptions=True
            )
            placed = sum(1 for r in results if r is True)
            logger.info(f"[{grid.symbol}] {side} placed: {placed}/{len(to_place)}")

    async def place_all_grids(self) -> None:
        """Place all grid orders for all symbols.

        Если grid уже восстановлен из биржи (есть PLACED уровни),
        размещаем только NEW уровни. Если чистый старт — все NEW.
        """
        for grid in self.grids.values():
            already_placed_buy = sum(1 for l in grid.buy_levels if l.status == "PLACED")
            already_placed_sell = sum(1 for l in grid.sell_levels if l.status == "PLACED")

            await self.place_side_orders(grid, "BUY")
            await self.place_side_orders(grid, "SELL")

            n_buy = sum(1 for l in grid.buy_levels if l.status == "PLACED")
            n_sell = sum(1 for l in grid.sell_levels if l.status == "PLACED")

            if already_placed_buy > 0 or already_placed_sell > 0:
                logger.info(
                    f"[{grid.symbol}] Grid placed: {n_buy} BUY, {n_sell} SELL "
                    f"(recovered: {already_placed_buy}B+{already_placed_sell}S already on exchange)"
                )
            else:
                logger.info(f"[{grid.symbol}] Grid placed: {n_buy} BUY, {n_sell} SELL")

    # ----- Fill detection via User Data Stream (primary) -----

    def _on_user_data(self, data: Any) -> None:
        """Callback for User Data Stream — push execution reports to async queue.

        Строго по официальному репозитарию binance-connector-python:
        - SDK парсит JSON → UserDataStreamEventsResponse (OneOf-обёртка)
        - .actual_instance содержит конкретную модель: ExecutionReport,
          OutboundAccountPosition, BalanceUpdate и т.д.
        - Для ExecutionReport поля: x=executionType, X=orderStatus,
          s=symbol, S=side, i=orderId, l=lastFillQty, L=lastFillPrice,
          n=commission, N=commissionAsset, z=cumFillQty, Z=cumQuoteQty
        """
        logger.debug(f"UDS callback: type={type(data).__name__}")

        actual = data
        if hasattr(data, "actual_instance") and data.actual_instance is not None:
            actual = data.actual_instance

        event_cls = type(actual).__name__
        logger.info(f"UDS event: {event_cls}")

        if event_cls == "ExecutionReport":
            exec_type = getattr(actual, "x", None)
            order_status = getattr(actual, "X", None)

            logger.info(
                f"UDS ExecutionReport: symbol={getattr(actual, 's', '?')} "
                f"side={getattr(actual, 'S', '?')} execType={exec_type} "
                f"status={order_status} orderId={getattr(actual, 'i', '?')} "
                f"clientOrderId={getattr(actual, 'c', '?')} "
                f"lastFillQty={getattr(actual, 'l', '?')} "
                f"lastFillPrice={getattr(actual, 'L', '?')}"
            )

            if exec_type in ("TRADE", "FILLED") and order_status in (
                "FILLED", "PARTIALLY_FILLED",
            ):
                if self._fill_events is not None:
                    try:
                        event_dict = actual.model_dump(by_alias=False)
                    except Exception:
                        event_dict = {
                            "e": "executionReport",
                            "x": exec_type,
                            "X": order_status,
                            "s": getattr(actual, "s", ""),
                            "S": getattr(actual, "S", ""),
                            "i": getattr(actual, "i", 0),
                            "l": getattr(actual, "l", "0"),
                            "L": getattr(actual, "L", "0"),
                            "z": getattr(actual, "z", "0"),
                            "Z": getattr(actual, "Z", "0"),
                            "n": getattr(actual, "n", "0"),
                            "N": getattr(actual, "N", ""),
                            "p": getattr(actual, "p", "0"),
                            "q": getattr(actual, "q", "0"),
                            "c": getattr(actual, "c", ""),
                        }
                    try:
                        self._fill_events.put_nowait(event_dict)
                    except asyncio.QueueFull:
                        logger.warning("Fill event queue full — dropping event")

        elif event_cls == "OutboundAccountPosition":
            # Строго по репозитарию:
            # OutboundAccountPosition.B = [{a: asset, f: free, l: locked}]
            # Приходит при ЛЮБОМ изменении баланса — и от бота, и от ручных торгов
            self._handle_outbound_account_position(actual)
        elif event_cls == "BalanceUpdate":
            # Строго по репозитарию:
            # BalanceUpdate: a=asset, d=delta (+/-), T=clearTime
            # Депозиты/выводы — логируем
            asset = getattr(actual, "a", "?")
            delta = getattr(actual, "d", "0")
            logger.info(f"UDS BalanceUpdate: asset={asset} delta={delta}")
        elif event_cls == "EventStreamTerminated":
            # Стрим умер — нужно переподписаться
            logger.warning("UDS: event stream terminated — scheduling resubscribe")
            self._uds_needs_resubscribe = True
        else:
            logger.debug(f"UDS: unknown event class={event_cls}")

    def _handle_outbound_account_position(self, actual: Any) -> None:
        """Обработать OutboundAccountPosition — обновить баланс из биржи.

        Строго по репозитарию binance-connector-python:
          - OutboundAccountPosition.B = список OutboundAccountPositionBInner
          - Поля: a=asset, f=free, l=locked

        Это событие приходит при ЛЮБОМ изменении баланса:
        - Наш ордер исполнен
        - Ручной ордер исполнен
        - Депозит/вывод
        - Комиссия списана

        Принцип: биржа — эталон. Обновляем usdt_free/coin_free из free,
        а usdt_in_buy_orders/coin_in_sell_orders пересчитываем из наших PLACED уровней.
        """
        n_symbols = max(1, len(self.grids)) if self.grids else 1
        ratio = self.config.balance_ratio

        # Извлекаем список балансов из события
        bals = getattr(actual, "B", None)
        if bals is None:
            return

        # Собираем балансы из события в словарь
        exchange_balances: Dict[str, tuple] = {}
        for b in bals:
            b = _unwrap(b)
            asset = _get_attr(b, "a", default="")
            free_str = _get_attr(b, "f", default="0")
            locked_str = _get_attr(b, "l", default="0")
            if asset:
                exchange_balances[asset] = (
                    _safe_decimal(free_str),
                    _safe_decimal(locked_str),
                )

        # Обновляем балансы для каждого grid
        for symbol, grid in self.grids.items():
            base = grid.info.base_asset

            usdt_free, usdt_locked = exchange_balances.get("USDT", (Decimal("0"), Decimal("0")))
            coin_free, coin_locked = exchange_balances.get(base, (Decimal("0"), Decimal("0")))

            # Наша доля от free баланса
            usdt_per = usdt_free * ratio / Decimal(str(n_symbols))
            coin_per = coin_free * ratio / Decimal(str(n_symbols))

            # Свободный баланс = наша доля - наши locked (в размещённых ордерах)
            old_usdt_free = grid.usdt_free
            old_coin_free = grid.coin_free

            grid.usdt_free = max(Decimal("0"), usdt_per - grid.usdt_in_buy_orders)
            grid.coin_free = max(Decimal("0"), coin_per - grid.coin_in_sell_orders)

            # Логируем если баланс существенно изменился (не от наших fills)
            usdt_diff = grid.usdt_free - old_usdt_free
            coin_diff = grid.coin_free - old_coin_free
            if abs(usdt_diff) > Decimal("0.01") or abs(coin_diff) > Decimal("0.001"):
                logger.info(
                    f"[{symbol}] Balance sync from exchange: "
                    f"USDT={grid.usdt_free:.2f} (was {old_usdt_free:.2f}, diff {usdt_diff:+.2f}), "
                    f"{base}={grid.coin_free:.4f} (was {old_coin_free:.4f}, diff {coin_diff:+.4f})"
                )

    async def _resubscribe_uds(self) -> bool:
        """Переподписать User Data Stream после EventStreamTerminated.

        Строго по репозитарию:
          - user_data_stream_subscribe_signature() → подписка с HMAC signing
          - res.stream → RequestStreamHandle → .on("message", cb)
        """
        if self.ws_api_conn is None:
            logger.warning("Cannot resubscribe UDS: WS API connection not available")
            return False

        # Отписываем старый стрим если есть
        if self._uds_stream is not None:
            try:
                await self._uds_stream.unsubscribe()
            except Exception:
                pass

        try:
            res = await self.ws_api_conn.user_data_stream_subscribe_signature()

            # Проверяем ответ
            uds_ok = False
            try:
                response = res.response
                resp_data = response.data() if response else None
                if resp_data:
                    status = _get_attr(resp_data, "status", default=0)
                    result = _get_attr(resp_data, "result", default=None)
                    if status == 200 and result is not None:
                        uds_ok = True
            except Exception:
                pass

            self._uds_stream = res.stream
            self._uds_stream.on("message", self._on_user_data)

            if uds_ok:
                logger.info("UDS resubscribed successfully after EventStreamTerminated")
            else:
                logger.warning("UDS resubscribe response unclear — may work, may not")

            return True

        except Exception as e:
            logger.error(f"UDS resubscribe failed: {e}")
            self._uds_stream = None
            return False

    async def _process_fill_events(self) -> None:
        """Process fill events with BATCHING — merge rapid fills into one grid shift.

        Ключевая оптимизация: когда приходят несколько fill-ов подряд (например,
        5 BUY fills за 1 секунду), НЕ делаем 5 отдельных shift-ов.
        Вместо этого:
        1. Дрейним ВСЕ pending fills из очереди
        2. Обновляем стейт для каждого fill-а (mark FILLED, update price/balance)
        3. Собираем какие стороны нуждаются в shift
        4. Берём lock и делаем ОДИН shift для каждой стороны
        5. Внутри shift loop: после завершения проверяем новые fills → повторяем

        Shift lock гарантирует что для одного символа НЕ может быть
        двух параллельных shift-ов (Bug 5 fix).
        """
        while self._running:
            try:
                try:
                    event = await asyncio.wait_for(
                        self._fill_events.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    if len(self._processed_fill_ids) > 10000:
                        self._processed_fill_ids.clear()
                    continue

                # Дрейним все pending fills — batch!
                fills = [event]
                while not self._fill_events.empty():
                    try:
                        fills.append(self._fill_events.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                # Обрабатываем все fills: обновляем стейт, собираем shifts
                pending_shifts: Dict[str, Set[str]] = {}
                for fill_data in fills:
                    result = self._update_fill_state(fill_data)
                    if result:
                        symbol, shift_side = result
                        if symbol not in pending_shifts:
                            pending_shifts[symbol] = set()
                        pending_shifts[symbol].add(shift_side)

                # Выполняем shift для каждого символа под lock-ом
                for symbol, sides in pending_shifts.items():
                    grid = self.grids.get(symbol)
                    if not grid:
                        continue
                    lock = self._shift_locks.setdefault(symbol, asyncio.Lock())
                    async with lock:
                        await self._execute_shift_loop(grid, sides)

            except Exception as e:
                logger.error(f"Fill event processing error: {e}")

    def _update_fill_state(self, data: dict) -> Optional[tuple]:
        """Update fill state SYNCHRONOUSLY — mark FILLED, update price & balance.

        НЕ делает grid shift — только обновляет стейт.
        Возвращает (symbol, shift_side) если нужен shift, иначе None.

        Поля ExecutionReport (строго по репозитарию):
          s=symbol, S=side(BUY/SELL), i=orderId, x=executionType,
          X=orderStatus, p=price, q=orderQty, l=lastFillQty, L=lastFillPrice,
          z=cumFillQty, Z=cumQuoteQty, n=commission, N=commissionAsset

        ЗАЩИТА ОТ ДВОЙНОЙ ОБРАБОТКИ:
          - _processed_fill_ids предотвращает двойную обработку одного fill-а
        """
        symbol = data.get("s", data.get("symbol", ""))
        order_id = data.get("i", data.get("orderId", 0))
        side = data.get("S", data.get("side", ""))
        order_status = data.get("X", data.get("orderStatus", ""))
        price_str = data.get("p", data.get("price", "0"))
        qty_str = data.get("q", data.get("quantity", "0"))
        last_filled_qty = data.get("l", data.get("lastFilledQuantity", "0"))
        commission_amount = data.get("n", data.get("commissionAmount", "0"))
        commission_asset = data.get("N", data.get("commissionAsset", ""))

        # Only process filled or partially filled orders
        if order_status not in ("FILLED", "PARTIALLY_FILLED"):
            return None

        # For PARTIALLY_FILLED, just log — we only shift on full fill
        if order_status == "PARTIALLY_FILLED":
            logger.debug(
                f"[{symbol}] Partial fill: {side} id={order_id} "
                f"filled={last_filled_qty} commission={commission_amount} {commission_asset}"
            )
            return None

        if symbol not in self.grids:
            return None

        int_order_id = int(order_id)

        # ЗАЩИТА 1: Глобальный набор обработанных fill-ов
        if int_order_id in self._processed_fill_ids:
            logger.debug(f"[{symbol}] Fill id={order_id} already processed (global) — skipping")
            return None

        # Помечаем как обработанный (sync — до любого await)
        self._processed_fill_ids.add(int_order_id)

        grid = self.grids[symbol]

        # ВАЖНО: сразу убираем из placed_order_ids (sync)
        grid.placed_order_ids.discard(int_order_id)

        # Find the grid level for this order
        filled_level = None
        filled_side = None
        for level in grid.buy_levels:
            if level.order_id == int_order_id:
                filled_level = level
                filled_side = "BUY"
                break
        if not filled_level:
            for level in grid.sell_levels:
                if level.order_id == int_order_id:
                    filled_level = level
                    filled_side = "SELL"
                    break

        if not filled_level:
            # Это не наш ордер — возможно ручная торговля или ордер из предыдущей сессии
            # Проверяем clientOrderId если доступен
            client_oid = data.get("c", data.get("clientOrderId", ""))
            if is_bot_order(client_oid):
                # Наш ордер но level потерян — что-то пошло не так
                logger.warning(
                    f"[{symbol}] Bot order fill id={order_id} cid={client_oid} "
                    f"not found in grid levels — level may have been lost during restart. "
                    f"Will refresh balances."
                )
            else:
                # Чужой ордер — ручная торговля. Не триггерим shift, но баланс
                # обновится через OutboundAccountPosition (приходит автоматически).
                logger.info(
                    f"[{symbol}] External order fill: {side} id={order_id} "
                    f"(not our order — balance will sync via OutboundAccountPosition)"
                )
            return None

        # Защита от повторной обработки (если poll уже обработал)
        if filled_level.status == "FILLED":
            logger.debug(f"[{symbol}] Fill id={order_id} level already FILLED — skipping")
            return None

        # Use actual fill data from the stream
        fill_price = _safe_decimal(price_str) if _safe_decimal(price_str) > 0 else filled_level.price
        fill_qty = _safe_decimal(last_filled_qty) if _safe_decimal(last_filled_qty) > 0 else filled_level.quantity
        actual_commission = _safe_decimal(commission_amount)

        source = "poll" if data.get("_source") == "poll" else "WS"

        logger.info(
            f"[{symbol}] ★ FILL ({source}): {filled_side} {fill_qty} @ {fill_price} "
            f"id={order_id} commission={actual_commission} {commission_asset}"
        )

        filled_level.status = "FILLED"

        fill_value = fill_price * fill_qty

        # Use actual commission from stream if available, otherwise estimate
        if actual_commission > 0:
            commission = actual_commission
        else:
            commission = fill_value * self.effective_commission

        # --- Комиссия: приводим всё к USDT-эквиваленту ---
        # Биржа списывает комиссию в разных активах:
        #   USDT → вычитаем из usdt_free напрямую
        #   base_asset (монета) → вычитаем из coin_free (монета)
        #   BNB → биржа списывает с BNB-баланса, НЕ трогаем USDT/coin
        # Но для commission_paid пересчитываем в USDT-эквивалент,
        # чтобы метрика была осмысленной (единая единица измерения).
        commission_usdt = Decimal("0")
        if commission_asset == "USDT":
            commission_usdt = commission
        elif commission_asset == grid.base_asset:
            commission_usdt = commission * fill_price  # монета → USDT по цене fill
        else:
            # BNB или другой актив — оцениваем по fill_price * fill_qty
            # как долю от объёма сделки (прикидка, точный курс BNB не знаем)
            # При BNB скидке: commission в BNB единицах, но нас интересует
            # USDT-эквивалент для статистики = fill_value * effective_commission
            commission_usdt = fill_value * self.effective_commission

        # Update balance tracking (sync)
        if filled_side == "BUY":
            grid.fills_buy += 1

            # BUY fill: потратили USDT (уже frozen в usdt_in_buy_orders),
            # получили монету (за вычетом комиссии если в монете)
            grid.usdt_in_buy_orders -= fill_value
            received_coin = fill_qty

            if commission_asset == "USDT":
                # Комиссия уже в USDT — USDT списаны биржей сверх fill_value
                grid.coin_free += fill_qty
                grid.usdt_free -= commission
            elif commission_asset == grid.base_asset:
                # Комиссия списана из полученной монеты
                received_coin = fill_qty - commission
                grid.coin_free += received_coin
            else:
                # Комиссия в BNB — биржа списала с BNB-баланса,
                # USDT и монету не трогаем — получаем полный fill_qty
                grid.coin_free += fill_qty

            grid.commission_paid_usdt += commission_usdt

            # Обновляем среднюю цену покупки для PnL
            grid.update_avg_buy(fill_price, received_coin)

            # Считаем сколько монеты реально доступно для SELL
            # (после вычета комиссии — это важно для корректного объёма SELL)

        elif filled_side == "SELL":
            grid.fills_sell += 1

            # SELL fill: отдали монету (уже frozen в coin_in_sell_orders),
            # получили USDT (за вычетом комиссии если в USDT)
            grid.coin_in_sell_orders -= fill_qty

            if commission_asset == "USDT":
                grid.usdt_free += fill_value - commission
            elif commission_asset == grid.base_asset:
                # Получили полную fill_value в USDT
                grid.usdt_free += fill_value
                # Комиссия из монеты — списываем из свободных монет
                grid.coin_free = max(Decimal("0"), grid.coin_free - commission)
            else:
                # Комиссия в BNB — биржа списала с BNB-баланса
                # USDT получаем полностью, монета не тронута
                grid.usdt_free += fill_value

            grid.commission_paid_usdt += commission_usdt

            # Считаем realized PnL
            pnl = grid.calc_realized_pnl(fill_price, fill_qty)
            grid.realized_pnl += pnl
            logger.info(
                f"[{symbol}] PnL: realized={pnl:.6f} total={grid.realized_pnl:.6f} "
                f"(SELL {fill_qty} @ {fill_price})"
            )

        # Обновляем счётчик инвентарного лимита
        # ВАЖНО: порядок — сначала снимаем старую паузу, потом ставим новую!
        # Иначе при SELL fill при BUY pause новый pause перезапишет старый
        # и снятие не сработает.
        inv_limit = self.config.inventory_limit
        if inv_limit > 0:
            # Шаг 1: Если fill на противоположной стороне — снимаем паузу
            if grid._inventory_pause == "BUY" and filled_side == "SELL":
                grid._consec_buy = 0
                grid._inventory_pause = None
                logger.info(f"[{symbol}] INVENTORY PAUSE lifted: BUY resumed (SELL fill received)")
            elif grid._inventory_pause == "SELL" and filled_side == "BUY":
                grid._consec_sell = 0
                grid._inventory_pause = None
                logger.info(f"[{symbol}] INVENTORY PAUSE lifted: SELL resumed (BUY fill received)")

            # Шаг 2: Обновляем счётчики
            if filled_side == "BUY":
                grid._consec_buy += 1
                grid._consec_sell = 0
            elif filled_side == "SELL":
                grid._consec_sell += 1
                grid._consec_buy = 0

            # Шаг 3: Если паузы нет — проверяем нужно ли установить
            if grid._inventory_pause is None:
                if filled_side == "BUY" and grid._consec_buy >= inv_limit:
                    grid._inventory_pause = "BUY"
                    logger.info(
                        f"[{symbol}] INVENTORY PAUSE: BUY paused "
                        f"({grid._consec_buy} consecutive BUY fills, limit={inv_limit})"
                    )
                elif filled_side == "SELL" and grid._consec_sell >= inv_limit:
                    grid._inventory_pause = "SELL"
                    logger.info(
                        f"[{symbol}] INVENTORY PAUSE: SELL paused "
                        f"({grid._consec_sell} consecutive SELL fills, limit={inv_limit})"
                    )

        # Определяем сторону для shift и anchor цену для rebuild
        shift_side = "SELL" if filled_side == "BUY" else "BUY"

        # Track anchor для rebuild — глубочайшая fill_price:
        # BUY fill  → shift SELL → берём самую НИЗКУЮ buy_fill_price
        # SELL fill → shift BUY  → берём самую ВЫСОКУЮ sell_fill_price
        if shift_side in grid._shift_anchor:
            old_anchor = grid._shift_anchor[shift_side]
            if filled_side == "BUY":
                # BUY fills идут вниз — берём самый глубокий (низкий)
                grid._shift_anchor[shift_side] = min(old_anchor, fill_price)
            else:
                # SELL fills идут вверх — берём самый высокий
                grid._shift_anchor[shift_side] = max(old_anchor, fill_price)
        else:
            grid._shift_anchor[shift_side] = fill_price

        return (symbol, shift_side)

    async def _execute_shift_loop(self, grid: SymbolGrid, initial_sides: Set[str]) -> None:
        """Execute grid shifts in a loop-until-stable pattern.

        После каждого shift проверяем, пришли ли новые fills.
        Если да — делаем ещё один shift. Если нет — стабильное состояние.

        Ключевые отличия от предыдущей версии:
        1. rebuild_sell_grid(anchor_price) / rebuild_buy_grid(anchor_price)
           — rebuild от fill_price, а не от volatile current_price
        2. Shift lock уже удерживается в _process_fill_events
           — параллельные shift-ы для одного символа невозможны
        3. После shift дрейним _fill_events для новых fills
        """
        sides_to_shift = initial_sides.copy()
        iteration = 0

        while sides_to_shift:
            iteration += 1
            sides_str = "+".join(sorted(sides_to_shift))
            q_size = self._fill_events.qsize() if self._fill_events else 0
            logger.info(
                f"[{grid.symbol}] Grid shift #{iteration}: {sides_str} "
                f"(center={grid.center_price}, anchors={dict(grid._shift_anchor)}, queue={q_size})"
            )

            # Cancel + rebuild + place для каждой стороны
            for side in sides_to_shift:
                await self.cancel_side_orders(grid, side)
                # Rebuild от _shift_anchor — глубочайшей fill_price,
                # а НЕ от устаревшего center_price (иначе gap растёт).
                anchor = grid._shift_anchor.pop(side, grid.center_price)
                if side == "SELL":
                    grid.rebuild_sell_grid(anchor)
                else:
                    grid.rebuild_buy_grid(anchor)
                logger.info(
                    f"[{grid.symbol}] {side} grid rebuilt from anchor={anchor} "
                    f"(was center={grid.center_price})"
                )
                await self.place_side_orders(grid, side)

            n_buy = sum(1 for l in grid.buy_levels if l.status == "PLACED")
            n_sell = sum(1 for l in grid.sell_levels if l.status == "PLACED")
            logger.info(f"[{grid.symbol}] Grid: {n_buy} BUY, {n_sell} SELL placed")
            logger.info(grid.status_line())

            # Дрейним _fill_events для fills пришедших во время shift
            sides_to_shift = set()
            while not self._fill_events.empty():
                try:
                    extra = self._fill_events.get_nowait()
                    result = self._update_fill_state(extra)
                    if result:
                        _, shift_side = result
                        sides_to_shift.add(shift_side)
                except asyncio.QueueEmpty:
                    break

            if sides_to_shift:
                logger.info(
                    f"[{grid.symbol}] More fills during shift — "
                    f"re-shifting {sides_str}"
                )

    # ----- Fill detection via polling (запасной механизм) -----

    async def poll_fills(self) -> None:
        """Detect fills by comparing placed_order_ids with open orders via WS API.

        Запасной механизм — вызывается редко (каждые 60с при работающем UDS).

        ВАЖНО: poll пушит fills в _fill_events очередь, проходит через
        тот же батчинг-пайплайн что и UDS fills.

        Bug 6 fix: Если fill уже обработан UDS (_processed_fill_ids),
        poll НЕ пушит его в очередь — значит НЕ триггерит grid shift.
        """
        for symbol, grid in list(self.grids.items()):
            if not grid.placed_order_ids:
                continue

            try:
                if self.ws_api_conn is not None:
                    resp = await self.ws_api_conn.open_orders_status(symbol=symbol)
                    data = _unwrap(resp.data())
                else:
                    resp = self.client.rest_api.get_open_orders(symbol=symbol)
                    data = _unwrap(resp.data())

                # Extract open order IDs
                open_ids: Set[int] = set()
                order_list = data
                if isinstance(data, dict):
                    for key in ("result", "orders", "data"):
                        val = data.get(key)
                        if isinstance(val, list):
                            order_list = val
                            break
                if not isinstance(order_list, list):
                    order_list = [order_list] if order_list else []
                for order in order_list:
                    order = _unwrap(order)
                    oid = _get_attr(order, "order_id", "orderId", default=None)
                    if oid:
                        open_ids.add(int(oid))

                filled_ids = grid.placed_order_ids - open_ids

                if filled_ids:
                    logger.info(
                        f"[{symbol}] Poll: {len(filled_ids)} filled orders detected "
                        f"(was {len(grid.placed_order_ids)} placed, {len(open_ids)} still open)"
                    )

                for oid in filled_ids:
                    # Bug 6 fix: Если UDS уже обработал — НЕ пушим в очередь
                    # Это полностью предотвращает re-triggering grid shift
                    if oid in self._processed_fill_ids:
                        logger.debug(
                            f"[{symbol}] Poll: fill {oid} already processed by UDS — "
                            f"skipping (no shift trigger)"
                        )
                        grid.placed_order_ids.discard(oid)
                        continue

                    # Пытаемся получить точные данные через order_status (WS API)
                    fill_price = None
                    fill_qty = None
                    fill_comm = "0"
                    fill_comm_asset = ""

                    if self.ws_api_conn is not None:
                        try:
                            resp = await self.ws_api_conn.order_status(
                                symbol=symbol, order_id=int(oid)
                            )
                            data = _unwrap(resp.data())
                            if isinstance(data, dict) and "error" not in data:
                                fill_price = _get_attr(data, "price", default=None)
                                fill_qty = _get_attr(data, "executed_qty", "executedQty", default=None)
                                fill_comm = _get_attr(data, "cummulative_quote_qty", "cummulativeQuoteQty", default="0")
                        except Exception:
                            pass  # fallback к данным из grid level

                    # Find which level was filled
                    filled_level = None
                    filled_side = None
                    for level in grid.buy_levels:
                        if level.order_id == oid:
                            filled_level = level
                            filled_side = "BUY"
                            break
                    if not filled_level:
                        for level in grid.sell_levels:
                            if level.order_id == oid:
                                filled_level = level
                                filled_side = "SELL"
                                break

                    if filled_level:
                        # Используем точные данные если есть, иначе из grid level
                        price = fill_price if fill_price else str(filled_level.price)
                        qty = fill_qty if fill_qty else str(filled_level.quantity)
                        event_dict = {
                            "s": symbol,
                            "S": filled_side,
                            "i": oid,
                            "X": "FILLED",
                            "x": "TRADE",
                            "p": price,
                            "q": qty,
                            "l": qty,
                            "L": price,
                            "n": fill_comm,
                            "N": fill_comm_asset,
                            "_source": "poll",
                        }
                        try:
                            self._fill_events.put_nowait(event_dict)
                            logger.debug(
                                f"[{symbol}] Poll: fill {oid} pushed to event queue "
                                f"({filled_side} @ {price})"
                            )
                        except asyncio.QueueFull:
                            logger.warning(
                                f"[{symbol}] Fill event queue full — dropping poll fill for {oid}"
                            )
                    else:
                        logger.warning(
                            f"[{symbol}] Filled order {oid} not found in grid levels — "
                            f"may have been from a previous session"
                        )

                    grid.placed_order_ids.discard(oid)

            except Exception as e:
                logger.debug(f"[{symbol}] Poll fills error: {e}")

    # ----- Balance refresh -----

    async def refresh_balances(self) -> None:
        """Периодическое обновление балансов через WS API.

        Используем только free из API, затем вычитаем наши locked
        (usdt_in_buy_orders / coin_in_sell_orders), чтобы получить
        актуальный свободный баланс для размещения новых ордеров.
        """
        try:
            balances = await self.fetch_balances()
            n_symbols = max(1, len(self.grids))

            for symbol, grid in self.grids.items():
                base = grid.info.base_asset
                usdt_free, usdt_locked = balances.get("USDT", (Decimal("0"), Decimal("0")))
                coin_free, coin_locked = balances.get(base, (Decimal("0"), Decimal("0")))

                usdt_per = usdt_free * self.config.balance_ratio / Decimal(str(n_symbols))
                coin_per = coin_free * self.config.balance_ratio / Decimal(str(n_symbols))

                # Наши locked (из размещённых ордеров) могут отличаться от API locked
                # из-за комиссий и частичных fills — берём максимум из обоих
                # Clamp: не может быть < 0 (рассинхрон tracking vs API)
                grid.usdt_free = max(Decimal("0"), usdt_per - grid.usdt_in_buy_orders)
                grid.coin_free = max(Decimal("0"), coin_per - grid.coin_in_sell_orders)

        except Exception as e:
            logger.debug(f"Balance refresh error: {e}")

    # ----- WebSocket event handlers -----

    def _on_book_ticker(self, data: Any) -> None:
        """Callback for bookTicker stream — update current price.

        Обновляет current_price (mid-price), но НЕ center_price!
        center_price обновляется только при fill/grid shift.
        """
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return
        elif hasattr(data, "__dict__") and not isinstance(data, dict):
            data = data.__dict__

        symbol = data.get("symbol", data.get("s", ""))
        bid_str = data.get("bid", data.get("b", "0"))
        ask_str = data.get("ask", data.get("a", "0"))

        if symbol not in self.grids:
            return

        bid = _safe_decimal(bid_str)
        ask = _safe_decimal(ask_str)

        if bid > 0 and ask > 0:
            mid = (bid + ask) / Decimal("2")
            self.grids[symbol].current_price = mid

    # ----- WebSocket API connection -----

    async def connect_ws_api(self) -> None:
        """Connect WS API for trading, account queries, and user data stream."""
        if self.client is None:
            raise RuntimeError("Client not initialized")

        try:
            self.ws_api_conn = await self.client.websocket_api.create_connection()
            logger.info("WS API connected")

            # Session logon — строго по репозитарию:
            # Пример: examples/websocket_api/Auth/session_logon.py
            # response.data() → SessionLogonResponse → check status + result
            # HMAC-ключи не поддерживаются для session.logon в Demo — это нормально,
            # потому что userDataStream.subscribe.signature подписывает каждый запрос
            # индивидуально через HMAC (отдельно от session.logon).
            try:
                response = await self.ws_api_conn.session_logon()
                data = _unwrap(response.data())

                # Проверяем успешность — по примеру из репозитария:
                # response.data() содержит status и result
                logon_ok = False
                if isinstance(data, dict):
                    status = data.get("status", 0)
                    if status == 200 and data.get("result") is not None:
                        logon_ok = True
                elif hasattr(data, "status") and data.status == 200:
                    logon_ok = True

                if logon_ok:
                    logger.info("WS API session logged on")
                else:
                    logger.warning(f"WS API session_logon response: {data}")
                    logger.info("WS API session logon not established — "
                                "will use per-request HMAC signing")
            except Exception as e:
                logger.warning(f"WS API session_logon failed (non-critical): {e}")
                logger.info("Per-request HMAC signing will be used instead")

            try:
                # Строго по примеру из репозитария:
                # examples/websocket_api/UserDataStream/user_data_stream_subscribe_signature.py
                # res.response → UserDataStreamSubscribeSignatureResponse
                #   .status == 200, .result.subscriptionId
                # res.stream → RequestStreamHandle (.on("message", cb))
                res = await self.ws_api_conn.user_data_stream_subscribe_signature()

                # Проверяем ответ — по примеру: res.response.data()
                uds_ok = False
                subscription_id = None
                try:
                    response = res.response
                    resp_data = response.data() if response else None
                    if resp_data:
                        # Pydantic model или dict
                        status = _get_attr(resp_data, "status", default=0)
                        result = _get_attr(resp_data, "result", default=None)
                        if status == 200 and result is not None:
                            uds_ok = True
                            subscription_id = _get_attr(result, "subscription_id", "subscriptionId", default="?")
                except Exception as check_err:
                    logger.debug(f"UDS response check error: {check_err}")

                self._uds_stream = res.stream
                self._uds_stream.on("message", self._on_user_data)

                if uds_ok:
                    logger.info(f"User Data Stream subscribed (HMAC signed), subscriptionId={subscription_id}")
                else:
                    # Может быть ошибка в res.response — логируем но продолжаем
                    logger.warning(f"UDS subscription response unclear — may work, may not")
                    logger.info("If no UDS events arrive, will use polling for fill detection")

            except Exception as e:
                logger.warning(
                    f"User Data Stream subscription failed: {e}. "
                    f"Will use polling for fill detection."
                )
                self._uds_stream = None

        except Exception as e:
            logger.error(f"WS API connection failed: {e}")
            logger.warning("Falling back to REST for all operations")
            self.ws_api_conn = None

    # ----- WebSocket Streams connection -----

    async def connect_ws_streams(self) -> None:
        """Connect WS Streams and subscribe to market data."""
        if self.client is None:
            raise RuntimeError("Client not initialized")

        try:
            self.ws_streams_conn = await self.client.websocket_streams.create_connection()

            for symbol in self.config.symbols:
                stream = await self.ws_streams_conn.book_ticker(symbol=symbol.lower())
                stream.on("message", self._on_book_ticker)
                logger.info(f"[{symbol}] bookTicker stream subscribed")
        except Exception as e:
            logger.warning(f"WS Streams connection failed: {e}")
            logger.warning("Will use REST for price updates as fallback")

    # ----- Health check / stats -----

    async def _ws_health_check(self) -> None:
        """Проверка здоровья WS-соединения и реконнект при необходимости.

        Строго по репозитарию binance-connector-python:
          - session_status() → SessionStatusResponseResult
            {apiKey, authorizedSince, connectedSince, userDataStream}
          - Если WS мёртв — пересоздаём через connect_ws_api()

        SDK делает автоматический реконнект для:
          - 23-часового рестарта Binance (запланированный)
          - serverShutdown event
        Но НЕ для:
          - Сетевого обрыва (ERROR/CLOSE) —这是我们检测的
        """
        if self.ws_api_conn is None:
            # WS API уже мёртв — пробуем пересоздать
            logger.warning("WS API connection is None — attempting reconnect")
            try:
                await self.connect_ws_api()
                self._ws_last_ok = time.monotonic()
                logger.info("WS API reconnected successfully")
            except Exception as e:
                logger.error(f"WS API reconnect failed: {e}")
            return

        try:
            # Строго по примеру: examples/websocket_api/Auth/session_status.py
            resp = await self.ws_api_conn.session_status()
            data = _unwrap(resp.data())

            # Проверяем что сессия жива
            if isinstance(data, dict) and "error" in data:
                raise ValueError(f"Session status error: {data['error']}")

            self._ws_last_ok = time.monotonic()

            # Проверяем что UDS ещё активен
            uds_active = False
            if isinstance(data, dict):
                uds_active = data.get("userDataStream", False)
            elif hasattr(data, "user_data_stream"):
                uds_active = data.user_data_stream

            if not uds_active and self._uds_stream is not None:
                logger.warning("WS session OK but UDS not active — resubscribing")
                self._uds_needs_resubscribe = True

        except Exception as e:
            time_since_ok = time.monotonic() - self._ws_last_ok
            logger.warning(
                f"WS health check failed: {e} "
                f"(last OK: {time_since_ok:.0f}s ago)"
            )

            # Если WS не отвечает дольше 2 интервалов — пересоздаём
            if time_since_ok > self._ws_check_interval * 2:
                logger.error(
                    f"WS API unresponsive for {time_since_ok:.0f}s — "
                    f"forcing reconnect"
                )
                try:
                    # Закрываем старое соединение
                    try:
                        await self.ws_api_conn.close_connection(close_session=True)
                    except Exception:
                        pass
                    self.ws_api_conn = None

                    # Пересоздаём
                    await self.connect_ws_api()
                    self._ws_last_ok = time.monotonic()
                    logger.info("WS API forced reconnect successful")
                except Exception as e2:
                    logger.error(f"WS API forced reconnect failed: {e2}")

    async def health_loop(self) -> None:
        """Periodic health check and status logging."""
        while self._running:
            await asyncio.sleep(self.config.health_interval)

            logger.info("=" * 50)
            logger.info("SPOT BOT STATUS")
            for grid in self.grids.values():
                logger.info(grid.status_line())
            total_pnl = sum(g.realized_pnl for g in self.grids.values())
            total_comm = sum(g.commission_paid_usdt for g in self.grids.values())
            logger.info(f"Total: pnl={total_pnl:.4f} commission={total_comm:.4f} net={total_pnl - total_comm:.4f}")
            logger.info(f"Commission: taker={self.taker_commission} BNB={'ON' if self.bnb_discount_enabled else 'OFF'} effective={self.effective_commission}")
            ws_mode = "WS API" if self.ws_api_conn else "REST"
            uds_sub_id = getattr(self._uds_stream, '_stream', '?') if self._uds_stream else None
            uds_mode = f"UDS(subId={uds_sub_id})" if self._uds_stream else "Polling"
            fill_q = self._fill_events.qsize() if self._fill_events else 0
            rl_stats = self._order_limiter.stats() if self._order_limiter else "N/A"
            logger.info(f"Transport: {ws_mode} | Fills: {uds_mode} | Queue: {fill_q}")
            logger.info(f"RateLimiter: {rl_stats}")
            logger.info("=" * 50)

    # ----- Main run loop -----

    async def run(self) -> None:
        """Start the bot."""
        self._running = True
        self._fill_events = asyncio.Queue(maxsize=1000)
        self._shift_locks = {sym: asyncio.Lock() for sym in self.config.symbols}
        self._processed_fill_ids = set()

        # Banner
        mode = self.config.mode.upper()
        logger.info("=" * 60)
        logger.info("  SPOT BOT — Market Maker (WebSocket)")
        logger.info(f"  Mode:    {mode}")
        logger.info(f"  Symbols: {', '.join(self.config.symbols)}")
        logger.info(f"  Step:    {self.config.grid_step_pct}%")
        logger.info(f"  Levels:  {self.config.grid_levels}")
        logger.info(f"  Order:   {self.config.grid_order_size_usdt} USDT (min={self.config.grid_min_order_usdt}, max={self.config.grid_max_order_usdt})")
        logger.info(f"  Asymmetry: {'ON' if self.config.asymmetry_enabled else 'OFF'}")
        inv_str = str(self.config.inventory_limit) if self.config.inventory_limit > 0 else "OFF"
        logger.info(f"  Inv limit: {inv_str}")
        logger.info(f"  Rate lim: {self.config.order_rate_limit} orders/sec")
        logger.info("=" * 60)

        # Step 1: Initialize client (REST + WS API + WS Streams)
        self.init_client()

        # Step 2: Connect WS API (trading + user data)
        await self.connect_ws_api()

        # Step 3: Setup symbols (uses WS API, fallback REST)
        await self.setup_symbols()

        if not self.grids:
            logger.error("No symbols set up — exiting")
            return

        # Step 4: Connect WS Streams (market data)
        await self.connect_ws_streams()

        # Step 5: Rate limiter — инициализируем ДО размещения ордеров!
        self._order_limiter = OrderRateLimiter(orders_per_second=self.config.order_rate_limit)
        logger.info(f"Rate limiter: {self.config.order_rate_limit} orders/sec "
                    f"(Binance limit: 10/sec, margin: {10 - self.config.order_rate_limit}/sec)")

        # Step 6: Place initial grid orders (via WS API) — rate-limited
        await self.place_all_grids()

        logger.info("Bot is running. Press Ctrl+C to stop.")

        # Step 7: Main loop — fill processing via UDS + periodic tasks + health monitoring

        last_balance_refresh = time.monotonic()
        last_poll_fills = time.monotonic()
        last_ws_health_check = time.monotonic()
        self._ws_last_ok = time.monotonic()

        poll_interval_no_uds = self.config.fill_poll_interval
        poll_interval_with_uds = max(self.config.balance_poll_interval, 60.0)

        try:
            fill_processor = asyncio.create_task(self._process_fill_events())

            while self._running:
                now = time.monotonic()

                # --- Poll fills (backup detection) ---
                poll_interval = poll_interval_no_uds if self._uds_stream is None else poll_interval_with_uds
                if now - last_poll_fills >= poll_interval:
                    await self.poll_fills()
                    last_poll_fills = now

                # --- Periodic balance refresh ---
                if now - last_balance_refresh >= self.config.balance_poll_interval:
                    await self.refresh_balances()
                    last_balance_refresh = now

                # --- UDS resubscribe if needed (EventStreamTerminated) ---
                if self._uds_needs_resubscribe:
                    self._uds_needs_resubscribe = False
                    logger.warning("Attempting UDS resubscribe...")
                    ok = await self._resubscribe_uds()
                    if ok:
                        logger.info("UDS resubscribe successful")
                    else:
                        logger.error("UDS resubscribe failed — falling back to polling")

                # --- WS health check ---
                if now - last_ws_health_check >= self._ws_check_interval:
                    last_ws_health_check = now
                    await self._ws_health_check()

                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Bot crashed: {e}", exc_info=True)
        finally:
            self._running = False
            fill_processor.cancel()
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up connections. Orders are LEFT on exchange for recovery.

        Принцип: биржа — эталон. При shutdown НЕ отменяем ордера.
        При следующем запуске бот восстановит состояние из ордеров на бирже.
        Ордера останутся жить на бирже и могут исполниться пока бот выключен.
        """
        logger.info("Shutting down... (orders left on exchange for recovery)")

        if self._uds_stream:
            try:
                await self._uds_stream.unsubscribe()
            except Exception:
                pass

        if self.ws_api_conn:
            try:
                await self.ws_api_conn.close_connection(close_session=True)
            except Exception:
                pass

        if self.ws_streams_conn:
            try:
                await self.ws_streams_conn.close_connection(close_session=True)
            except Exception:
                pass

        logger.info("Shutdown complete")


# ===================================================================
# Entry point
# ===================================================================

async def main():
    bot = SpotBot()
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped.")
