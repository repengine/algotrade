"""
CLI Monitoring Tool for AlgoStack.

Provides a command-line interface for monitoring the trading system.
"""

import asyncio
from datetime import datetime
from typing import Optional

import aiohttp
import click
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

console = Console()


class DashboardCLI:
    """CLI Dashboard for monitoring."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_system_info(self):
        """Get system information."""
        async with self.session.get(f"{self.api_url}/api/system/info") as resp:
            return await resp.json()

    async def get_positions(self):
        """Get current positions."""
        async with self.session.get(f"{self.api_url}/api/positions") as resp:
            return await resp.json()

    async def get_orders(self):
        """Get active orders."""
        async with self.session.get(f"{self.api_url}/api/orders") as resp:
            return await resp.json()

    async def get_performance(self):
        """Get performance metrics."""
        async with self.session.get(f"{self.api_url}/api/performance") as resp:
            return await resp.json()

    async def get_strategies(self):
        """Get strategy information."""
        async with self.session.get(f"{self.api_url}/api/strategies") as resp:
            return await resp.json()

    def create_system_panel(self, info: dict) -> Panel:
        """Create system information panel."""
        status = info.get("status", "unknown")
        mode = info.get("mode", "unknown")
        uptime = info.get("uptime_seconds", 0)

        # Format uptime
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Status color
        status_color = "green" if status == "running" else "red"

        content = f"""[{status_color}]Status:[/{status_color}] {status.upper()}
Mode: {mode.upper()}
Uptime: {uptime_str}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

        return Panel(content, title="System Info", border_style="blue")

    def create_positions_table(self, positions: list) -> Table:
        """Create positions table."""
        table = Table(title="Positions", show_header=True, header_style="bold magenta")
        table.add_column("Symbol", style="cyan", no_wrap=True)
        table.add_column("Quantity", justify="right")
        table.add_column("Avg Cost", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L %", justify="right")

        for pos in positions:
            pnl = pos.get("unrealized_pnl", 0)
            pnl_pct = pos.get("pnl_percentage", 0)
            pnl_color = "green" if pnl >= 0 else "red"

            table.add_row(
                pos.get("symbol", ""),
                str(pos.get("quantity", 0)),
                f"${pos.get('average_cost', 0):.2f}",
                f"${pos.get('current_price', 0):.2f}",
                f"[{pnl_color}]${pnl:,.2f}[/{pnl_color}]",
                f"[{pnl_color}]{pnl_pct:+.2f}%[/{pnl_color}]",
            )

        if not positions:
            table.add_row("[dim]No positions[/dim]", "", "", "", "", "")

        return table

    def create_orders_table(self, orders: list) -> Table:
        """Create orders table."""
        table = Table(
            title="Active Orders", show_header=True, header_style="bold magenta"
        )
        table.add_column("Order ID", style="cyan", no_wrap=True)
        table.add_column("Symbol")
        table.add_column("Side")
        table.add_column("Qty", justify="right")
        table.add_column("Type")
        table.add_column("Price", justify="right")
        table.add_column("Status")

        for order in orders[:10]:  # Show max 10 orders
            side_color = "green" if order.get("side") == "buy" else "red"

            table.add_row(
                order.get("order_id", "")[:8] + "...",
                order.get("symbol", ""),
                f"[{side_color}]{order.get('side', '').upper()}[/{side_color}]",
                str(order.get("quantity", 0)),
                order.get("order_type", ""),
                (
                    f"${order.get('limit_price', 0):.2f}"
                    if order.get("limit_price")
                    else "MKT"
                ),
                order.get("status", ""),
            )

        if not orders:
            table.add_row("[dim]No active orders[/dim]", "", "", "", "", "", "")

        return table

    def create_performance_panel(self, metrics: dict) -> Panel:
        """Create performance metrics panel."""
        total_value = metrics.get("total_value", 0)
        cash = metrics.get("cash", 0)
        total_pnl = metrics.get("total_pnl", 0)
        total_pnl_pct = metrics.get("total_pnl_percentage", 0)

        pnl_color = "green" if total_pnl >= 0 else "red"

        content = f"""Total Value: ${total_value:,.2f}
Cash: ${cash:,.2f}
Positions: ${total_value - cash:,.2f}

Total P&L: [{pnl_color}]${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)[/{pnl_color}]
Win Rate: {metrics.get('win_rate', 0)*100:.1f}%
Total Trades: {metrics.get('trades_total', 0)}"""

        return Panel(content, title="Performance", border_style="green")

    def create_strategies_panel(self, strategies: list) -> Panel:
        """Create strategies panel."""
        content = []

        for strategy in strategies:
            status_color = "green" if strategy.get("status") == "active" else "yellow"
            content.append(
                f"[{status_color}]● {strategy.get('name', 'Unknown')}[/{status_color}] "
                f"({strategy.get('id', '')})"
            )
            content.append(f"  Symbols: {', '.join(strategy.get('symbols', []))}")
            content.append(f"  Orders: {strategy.get('orders_placed', 0)}")
            content.append("")

        if not content:
            content = ["[dim]No strategies running[/dim]"]

        return Panel(
            "\n".join(content).strip(), title="Strategies", border_style="yellow"
        )

    async def create_dashboard(self) -> Layout:
        """Create full dashboard layout."""
        # Fetch all data
        try:
            system_info = await self.get_system_info()
            positions = await self.get_positions()
            orders = await self.get_orders()
            performance = await self.get_performance()
            strategies = await self.get_strategies()
        except Exception as e:
            error_panel = Panel(
                f"[red]Error connecting to API: {e}[/red]",
                title="Connection Error",
                border_style="red",
            )
            layout = Layout()
            layout.update(error_panel)
            return layout

        # Create layout
        layout = Layout()

        # Split into rows
        layout.split_column(
            Layout(name="header", size=7),
            Layout(name="body"),
            Layout(name="footer", size=10),
        )

        # Header row
        layout["header"].split_row(
            Layout(self.create_system_panel(system_info), name="system"),
            Layout(self.create_performance_panel(performance), name="performance"),
        )

        # Body - positions and orders
        layout["body"].split_row(
            Layout(self.create_positions_table(positions), name="positions"),
            Layout(self.create_orders_table(orders), name="orders"),
        )

        # Footer - strategies
        layout["footer"].update(self.create_strategies_panel(strategies))

        return layout


@click.group()
def cli():
    """AlgoStack CLI Monitoring Tool."""
    pass


@cli.command()
@click.option("--api-url", default="http://localhost:8000", help="API URL")
@click.option("--refresh", default=1, help="Refresh interval in seconds")
def monitor(api_url: str, refresh: int):
    """Live monitoring dashboard."""

    async def run_monitor():
        async with DashboardCLI(api_url) as dashboard:
            with Live(auto_refresh=False, refresh_per_second=1) as live:
                while True:
                    try:
                        layout = await dashboard.create_dashboard()
                        live.update(layout, refresh=True)
                        await asyncio.sleep(refresh)
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        console.print(f"[red]Error: {e}[/red]")
                        await asyncio.sleep(5)

    asyncio.run(run_monitor())


@cli.command()
@click.option("--api-url", default="http://localhost:8000", help="API URL")
def positions(api_url: str):
    """Show current positions."""

    async def show_positions():
        async with DashboardCLI(api_url) as dashboard:
            try:
                positions = await dashboard.get_positions()
                table = dashboard.create_positions_table(positions)
                console.print(table)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    asyncio.run(show_positions())


@cli.command()
@click.option("--api-url", default="http://localhost:8000", help="API URL")
def orders(api_url: str):
    """Show active orders."""

    async def show_orders():
        async with DashboardCLI(api_url) as dashboard:
            try:
                orders = await dashboard.get_orders()
                table = dashboard.create_orders_table(orders)
                console.print(table)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    asyncio.run(show_orders())


@cli.command()
@click.option("--api-url", default="http://localhost:8000", help="API URL")
def performance(api_url: str):
    """Show performance metrics."""

    async def show_performance():
        async with DashboardCLI(api_url) as dashboard:
            try:
                metrics = await dashboard.get_performance()
                panel = dashboard.create_performance_panel(metrics)
                console.print(panel)
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    asyncio.run(show_performance())


@cli.command()
@click.option("--api-url", default="http://localhost:8000", help="API URL")
@click.argument("action", type=click.Choice(["stop", "pause", "resume"]))
@click.confirmation_option(prompt="Are you sure?")
def control(api_url: str, action: str):
    """Control system state."""

    async def send_control():
        async with aiohttp.ClientSession() as session:
            try:
                data = {"action": action, "confirm": True}
                async with session.post(
                    f"{api_url}/api/system/control", json=data
                ) as resp:
                    result = await resp.json()
                    console.print(f"[green]Success: {result}[/green]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    asyncio.run(send_control())


if __name__ == "__main__":
    cli()
