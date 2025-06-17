"""Data and execution adapters for various providers."""

__all__ = ["IBKRAdapter", "Contract", "Order", "OrderType", "OrderSide", "TimeInForce"]

# Lazy imports to avoid circular dependencies and missing modules
def __getattr__(name):
    if name == "IBKRAdapter":
        from .ibkr_adapter import IBKRAdapter
        return IBKRAdapter
    elif name == "Contract":
        from .ibkr_adapter import Contract
        return Contract
    elif name == "Order":
        from .ibkr_adapter import Order
        return Order
    elif name == "OrderType":
        from .ibkr_adapter import OrderType
        return OrderType
    elif name == "OrderSide":
        from .ibkr_adapter import OrderSide
        return OrderSide
    elif name == "TimeInForce":
        from .ibkr_adapter import TimeInForce
        return TimeInForce
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
