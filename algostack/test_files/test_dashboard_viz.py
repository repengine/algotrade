#!/usr/bin/env python3
"""
Test script to verify dashboard visualization fixes
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Generate sample data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
price_start = 100
prices = price_start + np.cumsum(np.random.randn(len(dates)) * 2)

# Create sample strategy returns
strategy_returns = np.random.randn(len(dates)) * 0.02  # 2% daily volatility
cumulative_returns = (1 + strategy_returns).cumprod()

# Create figure with proper secondary y-axis configuration
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=('Price & Strategy Performance', 'Position'),
    row_heights=[0.7, 0.3],
    specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
)

# Plot price on primary y-axis
fig.add_trace(
    go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color='black', width=2)
    ),
    row=1, col=1,
    secondary_y=False
)

# Plot strategy returns on secondary y-axis
strategy_pct = (cumulative_returns - 1) * 100
fig.add_trace(
    go.Scatter(
        x=dates,
        y=strategy_pct,
        mode='lines',
        name='Strategy Returns',
        line=dict(color='blue', width=2)
    ),
    row=1, col=1,
    secondary_y=True
)

# Add some trade markers
trade_indices = [50, 100, 150, 200, 250]
for idx in trade_indices:
    fig.add_trace(
        go.Scatter(
            x=[dates[idx]],
            y=[prices[idx]],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green',
                line=dict(width=2, color='white')
            ),
            showlegend=False
        ),
        row=1, col=1,
        secondary_y=False
    )

# Update axes
fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="Returns (%)", row=1, col=1, secondary_y=True)

fig.update_layout(
    height=600,
    title="Test: Dual Y-Axis Visualization",
    hovermode='x unified'
)

# Print ranges to verify
print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
print(f"Returns range: {strategy_pct.min():.2f}% - {strategy_pct.max():.2f}%")
print("\nIf visualization works correctly:")
print("- Price should show on left y-axis with full range (not -1 to 1)")
print("- Strategy returns should show on right y-axis as percentage")
print("- Trade markers should appear on the price line")

# Save as HTML for inspection
fig.write_html("/home/republic/algotrade/algostack/test_viz.html")
print("\nVisualization saved to test_viz.html")