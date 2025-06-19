# AlgoTrade Phase 2: API and Integration Implementation Plan

## 🎯 PHASE 2 OVERVIEW
**Timeline**: Days 4-5 (2 days)
**Focus**: API endpoints and Data Source integration
**Prime Directive**: Enable system monitoring, control, and reliable market data access

## 📊 PHASE 1 COMPLETION STATUS ✅
All Phase 1 critical safety features have been successfully implemented:
- ✅ LiveTradingEngine methods (process_market_data, collect_signals, _update_portfolio)
- ✅ OrderManager critical methods (add_order, update_order_status)
- ✅ Risk Management enhancements (type handling, correlation checks, pre-trade validation)

## 🚀 PHASE 2 OBJECTIVES

### Day 4: API Endpoint Implementation
**Goal**: Full REST API for system monitoring and control
**Pillar Focus**: OPERATIONAL STABILITY & VERIFIABLE CORRECTNESS

### Day 5: Data Source Integration
**Goal**: Reliable market data fetching with failover mechanisms
**Pillar Focus**: OPERATIONAL STABILITY & PROFIT GENERATION

## 📡 DAY 4: API ENDPOINT IMPLEMENTATION

### 4.1 Missing Endpoints Analysis
Based on test failures, the following endpoints need implementation:

#### Core System Endpoints
1. **`GET /health`** - System health check
   - Returns: `{"status": "healthy"|"degraded"|"unhealthy", "timestamp": "...", "details": {...}}`
   - Checks: Database connection, broker connection, data feed status
   - Response time target: <100ms

2. **`GET /status`** - Trading system status
   - Returns: Full system state including engine status, active strategies, risk metrics
   - Includes: Market hours check, circuit breaker status, last update times
   - Response time target: <200ms

#### Trading Operations Endpoints
3. **`GET /positions`** - Current positions
   - Returns: All open positions with P&L, risk metrics
   - Filters: by symbol, strategy_id, date range
   - Includes: Unrealized P&L, position age, risk contribution

4. **`GET /positions/{position_id}`** - Single position details
   - Returns: Full position details including history
   - Includes: Entry/exit fills, P&L curve, risk metrics

5. **`GET /orders`** - Order management
   - Returns: All orders (pending, filled, cancelled)
   - Filters: by status, symbol, strategy_id, date range
   - Pagination: Required (default 100 per page)

6. **`POST /orders`** - Submit new order
   - Payload: `{"symbol": "...", "side": "...", "quantity": ..., "order_type": "...", ...}`
   - Validation: Risk checks, position limits, market hours
   - Returns: Order confirmation with order_id

7. **`PUT /orders/{order_id}`** - Modify order
   - Allowed modifications: quantity, limit price (pending orders only)
   - Validation: Risk re-check on modifications

8. **`DELETE /orders/{order_id}`** - Cancel order
   - Only for pending orders
   - Returns: Cancellation confirmation

#### Performance & Analytics Endpoints
9. **`GET /performance`** - Performance metrics
   - Returns: P&L, Sharpe, Drawdown, Win rate, etc.
   - Timeframes: 1D, 1W, 1M, 3M, YTD, All
   - Grouping: by strategy, symbol, sector

10. **`GET /performance/pnl`** - P&L timeseries
    - Returns: Timestamped P&L data
    - Granularity: 1m, 5m, 1h, 1d
    - Includes: Realized, unrealized, fees

#### Risk Management Endpoints
11. **`GET /risk/metrics`** - Current risk metrics
    - Returns: VaR, portfolio volatility, correlation matrix
    - Includes: Limit usage, risk warnings

12. **`GET /risk/limits`** - Risk limit configuration
    - Returns: All configured risk limits and current usage

13. **`PUT /risk/limits`** - Update risk limits
    - Requires: Admin authentication
    - Validation: Sanity checks on new limits

### 4.2 Implementation Strategy

#### Step 1: Router Setup
```python
# src/api/app.py
from fastapi import FastAPI, HTTPException, Depends
from src.api.routers import health, trading, performance, risk

app = FastAPI(title="AlgoTrade API", version="2.0.0")

# Include routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(trading.router, prefix="/api/v1")
app.include_router(performance.router, prefix="/api/v1")
app.include_router(risk.router, prefix="/api/v1")
```

#### Step 2: Dependency Injection
```python
# src/api/dependencies.py
async def get_trading_engine() -> LiveTradingEngine:
    """Get the singleton trading engine instance"""
    
async def get_risk_manager() -> RiskManager:
    """Get the risk manager instance"""
    
async def require_auth(token: str = Header(...)) -> User:
    """Validate API authentication"""
```

#### Step 3: Error Handling
```python
# src/api/exceptions.py
@app.exception_handler(TradingError)
async def trading_error_handler(request: Request, exc: TradingError):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc), "type": "trading_error"}
    )

@app.exception_handler(RiskLimitExceeded)
async def risk_limit_handler(request: Request, exc: RiskLimitExceeded):
    return JSONResponse(
        status_code=403,
        content={"error": str(exc), "type": "risk_limit_exceeded"}
    )
```

#### Step 4: Rate Limiting
```python
# src/api/middleware.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Different limits for different endpoints
@app.get("/orders")
@limiter.limit("100/minute")
async def get_orders():
    pass

@app.post("/orders")
@limiter.limit("10/minute")  # Stricter for order submission
async def create_order():
    pass
```

### 4.3 Testing Requirements
Each endpoint must have:
1. **Unit tests** - Test business logic in isolation
2. **Integration tests** - Test with real engine/database
3. **Load tests** - Ensure performance under load
4. **Error case tests** - All failure scenarios

## 💾 DAY 5: DATA SOURCE INTEGRATION

### 5.1 YFinanceFetcher Implementation

#### Core Interface Implementation
```python
# src/data/sources/yfinance_fetcher.py
class YFinanceFetcher(DataFetcher):
    async def fetch(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical data with validation and error handling"""
        
    async def fetch_realtime(
        self,
        symbols: List[str]
    ) -> Dict[str, MarketData]:
        """Fetch real-time quotes with circuit breaker detection"""
        
    def validate_data(
        self,
        data: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """Validate fetched data for completeness and sanity"""
```

### 5.2 Failover Mechanism

#### Multi-Source Architecture
```python
# src/data/sources/composite_fetcher.py
class CompositeFetcher(DataFetcher):
    def __init__(self, sources: List[DataFetcher]):
        self.sources = sources
        self.health_monitor = SourceHealthMonitor()
    
    async def fetch(self, symbols: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """Try each source in priority order until success"""
        for source in self.get_healthy_sources():
            try:
                data = await source.fetch(symbols, **kwargs)
                self.health_monitor.record_success(source)
                return data
            except Exception as e:
                self.health_monitor.record_failure(source, e)
                continue
        raise DataFetchError("All data sources failed")
```

### 5.3 Data Validation Pipeline

#### Validation Steps
1. **Completeness Check**
   - No missing critical fields (OHLCV)
   - No large gaps in time series
   
2. **Sanity Checks**
   - Prices > 0
   - Volume >= 0
   - High >= Low
   - High >= Open, Close
   - Low <= Open, Close
   
3. **Anomaly Detection**
   - Price movements > 50% flagged
   - Volume spikes > 10x average flagged
   - Stale data detection (no changes)

### 5.4 Caching Strategy
```python
# src/data/cache/redis_cache.py
class RedisDataCache:
    async def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached data if fresh enough"""
        
    async def set(self, key: str, data: pd.DataFrame, ttl: int = 300):
        """Cache data with TTL"""
        
    def make_key(self, symbol: str, interval: str, date: datetime) -> str:
        """Generate cache key"""
```

## 🧪 TESTING STRATEGY

### API Testing Matrix
| Endpoint | Unit | Integration | Load | Security |
|----------|------|-------------|------|----------|
| /health | ✓ | ✓ | ✓ | ✓ |
| /status | ✓ | ✓ | ✓ | ✓ |
| /positions | ✓ | ✓ | ✓ | ✓ |
| /orders | ✓ | ✓ | ✓ | ✓ |
| /performance | ✓ | ✓ | ✓ | ✓ |
| /risk/* | ✓ | ✓ | ✓ | ✓ |

### Data Source Testing
1. **Mock Data Tests** - Test with synthetic data
2. **Historical Data Tests** - Test with known historical scenarios
3. **Real-time Tests** - Test with live market data (dev environment)
4. **Failover Tests** - Test source switching
5. **Validation Tests** - Test data quality checks

## 📋 IMPLEMENTATION CHECKLIST

### Day 4: API Endpoints
- [ ] Create router structure and organize endpoints
- [ ] Implement health and status endpoints
- [ ] Implement position management endpoints
- [ ] Implement order management endpoints
- [ ] Implement performance tracking endpoints
- [ ] Implement risk management endpoints
- [ ] Add authentication and rate limiting
- [ ] Write comprehensive API tests
- [ ] Update API documentation

### Day 5: Data Integration
- [ ] Implement YFinanceFetcher.fetch() method
- [ ] Add data validation pipeline
- [ ] Implement failover mechanism
- [ ] Add caching layer
- [ ] Implement real-time quote fetching
- [ ] Write data source tests
- [ ] Test failover scenarios
- [ ] Performance test data pipeline

## 🎯 SUCCESS CRITERIA

### API Success Metrics
1. **All endpoints return < 500ms** (99th percentile)
2. **100% test coverage** on all endpoints
3. **Graceful error handling** for all failure modes
4. **Rate limiting prevents abuse**
5. **API documentation is complete**

### Data Integration Success Metrics
1. **Data fetching success rate > 99.9%**
2. **Failover happens < 1 second**
3. **Invalid data caught 100% of time**
4. **Cache hit rate > 80%** for historical data
5. **Real-time quotes < 100ms latency**

## ⚠️ RISK CONSIDERATIONS

### API Risks
- **Security**: Implement proper authentication before production
- **Performance**: Monitor endpoint response times
- **Stability**: Circuit breakers for downstream services
- **Compliance**: Audit trail for all operations

### Data Risks
- **Quality**: Bad data = bad trades
- **Latency**: Stale data = missed opportunities
- **Cost**: API rate limits and paid tiers
- **Legal**: Ensure data usage compliance

## 🔄 ROLLBACK PLAN
If issues arise:
1. **API**: Revert to previous version, maintain old endpoints
2. **Data**: Fall back to single source temporarily
3. **Both**: Full system halt if data integrity compromised

## 📈 NEXT STEPS (Phase 3 Preview)
After Phase 2 completion:
- Phase 3 will focus on test alignment and async pattern fixes
- Ensure all tests reflect production scenarios
- Fix remaining edge cases and race conditions

Remember: **Every endpoint and data point must serve the Four Pillars!**