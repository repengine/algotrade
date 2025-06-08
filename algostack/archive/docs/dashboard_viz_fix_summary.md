# Dashboard Visualization Fix Summary

## Problem
The dashboard was showing price data constrained to a -1 to 1 range instead of the full price range, and strategy returns were not displaying properly on a secondary y-axis.

## Root Cause
The `make_subplots` function was not configured with `specs` parameter to enable secondary y-axis support, and traces were being added with incorrect parameters.

## Solution Applied

### 1. Fixed Subplot Creation
```python
# Before:
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=(f'{symbol} Price & Strategy Performance', 'Strategy Signals'),
    row_heights=[0.7, 0.3]
)

# After:
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=(f'{symbol} Price & Strategy Performance', 'Strategy Signals'),
    row_heights=[0.7, 0.3],
    specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
)
```

### 2. Fixed Trace Addition
```python
# Before:
fig.add_trace(trace, row=1, col=1)  # No secondary_y parameter
fig.add_trace(trace_with_yaxis2, row=1, col=1)  # Using yaxis='y2'

# After:
fig.add_trace(price_trace, row=1, col=1, secondary_y=False)  # Price on primary
fig.add_trace(returns_trace, row=1, col=1, secondary_y=True)  # Returns on secondary
```

### 3. Fixed Y-Axis Updates
```python
# Before:
fig.update_yaxes(title_text="Price ($)", row=1, col=1)
fig.update_yaxes(title_text="Returns (%)", secondary_y=True, row=1, col=1)
fig.update_layout(yaxis2=dict(overlaying='y', side='right'))

# After:
fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="Returns (%)", row=1, col=1, secondary_y=True)
# No need for manual yaxis2 configuration
```

## Results
- Price data now displays with full range on the left y-axis
- Strategy returns display as percentages on the right y-axis
- Trade markers (buy/sell points) correctly appear on the price line
- Both axes scale independently for optimal visualization

## Files Modified
- `/home/republic/algotrade/algostack/dashboard_debug.py` - Fixed and tested
- `/home/republic/algotrade/algostack/dashboard.py` - Updated with fixes

## Verification
Created `test_dashboard_viz.py` to verify the dual y-axis functionality works correctly with sample data.