"""
API Models for AlgoStack Monitoring Dashboard.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class SystemStatus(str, Enum):
    """System status enumeration."""

    OFFLINE = "offline"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"


class StrategyStatus(str, Enum):
    """Strategy status enumeration."""

    INACTIVE = "inactive"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


class OrderStatus(str, Enum):
    """Order status enumeration."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# Response Models


class SystemInfo(BaseModel):
    """System information response."""

    status: SystemStatus
    mode: str
    uptime_seconds: float
    start_time: Optional[datetime]
    version: str = "1.0.0"


class StrategyInfo(BaseModel):
    """Strategy information."""

    id: str
    name: str
    status: StrategyStatus
    symbols: list[str]
    parameters: dict[str, Any]
    signals_generated: int
    orders_placed: int
    last_signal_time: Optional[datetime]


class PositionInfo(BaseModel):
    """Position information."""

    symbol: str
    quantity: int
    average_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    pnl_percentage: float


class OrderInfo(BaseModel):
    """Order information."""

    order_id: str
    symbol: str
    side: str
    quantity: int
    order_type: str
    status: OrderStatus
    limit_price: Optional[float]
    stop_price: Optional[float]
    filled_quantity: int
    average_fill_price: float
    submitted_at: datetime
    filled_at: Optional[datetime]
    strategy_id: Optional[str]


class PerformanceMetrics(BaseModel):
    """Performance metrics."""

    total_value: float
    cash: float
    positions_value: float
    daily_pnl: float
    daily_pnl_percentage: float
    total_pnl: float
    total_pnl_percentage: float
    sharpe_ratio: Optional[float]
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades_today: int
    trades_total: int


class RiskMetrics(BaseModel):
    """Risk metrics."""

    current_leverage: float
    max_leverage: float
    var_95: float  # Value at Risk
    position_concentration: dict[str, float]
    sector_exposure: dict[str, float]
    correlation_risk: float
    margin_usage: float
    buying_power: float


class AlertInfo(BaseModel):
    """Alert information."""

    id: str
    timestamp: datetime
    level: str  # info, warning, error, critical
    category: str  # trade, risk, system, data
    message: str
    details: Optional[dict[str, Any]]
    acknowledged: bool = False


class SignalInfo(BaseModel):
    """Trading signal information."""

    timestamp: datetime
    strategy_id: str
    symbol: str
    direction: int  # 1 for buy, -1 for sell
    strength: float
    metadata: dict[str, Any] = {}


class TradeInfo(BaseModel):
    """Completed trade information."""

    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str  # long or short
    pnl: float
    pnl_percentage: float
    commission: float
    strategy_id: Optional[str]
    duration_minutes: int


# Request Models


class StrategyCommand(BaseModel):
    """Strategy control command."""

    action: str = Field(..., pattern="^(enable|disable|reset)$")
    strategy_id: str
    parameters: Optional[dict[str, Any]] = None


class OrderCommand(BaseModel):
    """Manual order command."""

    symbol: str
    side: str = Field(..., pattern="^(buy|sell)$")
    quantity: int = Field(..., gt=0)
    order_type: str = Field(default="market", pattern="^(market|limit|stop|stop_limit)$")
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = Field(default="day", pattern="^(day|gtc|ioc|fok)$")

    @model_validator(mode='after')
    def validate_order_prices(self) -> 'OrderCommand':
        """Validate that limit/stop orders have appropriate prices."""
        if self.order_type in ['limit', 'stop_limit'] and self.limit_price is None:
            raise ValueError(f"{self.order_type} order requires limit_price")
        if self.order_type in ['stop', 'stop_limit'] and self.stop_price is None:
            raise ValueError(f"{self.order_type} order requires stop_price")
        return self


class RiskOverride(BaseModel):
    """Risk limit override."""

    parameter: str
    value: float
    duration_minutes: Optional[int] = None
    reason: str


class SystemCommand(BaseModel):
    """System control command."""

    action: str = Field(..., pattern="^(start|stop|pause|resume|emergency_stop)$")
    confirm: bool = False


# WebSocket Messages


class WSMessage(BaseModel):
    """WebSocket message base."""

    type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Any


class WSSubscription(BaseModel):
    """WebSocket subscription request."""

    action: str = Field(..., pattern="^(subscribe|unsubscribe)$")
    channels: list[str]  # positions, orders, signals, alerts, metrics


# Dashboard Configuration


class DashboardConfig(BaseModel):
    """Dashboard configuration."""

    refresh_interval: int = 1000  # milliseconds
    chart_periods: list[str] = ["1D", "1W", "1M", "3M", "1Y"]
    visible_strategies: list[str] = []
    alert_filters: list[str] = []
    theme: str = "dark"
