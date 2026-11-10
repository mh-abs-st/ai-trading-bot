"""Command line: fetch data, backtest, get recommendations, run the paper bot."""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from aitrader.backtest.engine import run_backtest
from aitrader.bot.trader import Trader
from aitrader.config import Settings, load_settings
from aitrader.data.source import (
    BinanceSource,
    CsvSource,
    MarketDataSource,
    SyntheticSource,
    save_csv,
)
from aitrader.execution.broker import PaperBroker
from aitrader.recommend.advisor import recommend
from aitrader.risk.manager import RiskManager
from aitrader.strategies import create_strategy

DISCLAIMER = (
    "These figures come from simulated trades on past data. They are not financial "
    "advice and do not predict future results."
)


def _source(args: argparse.Namespace) -> MarketDataSource:
    if args.data:
        return CsvSource(args.data)
    if args.synthetic:
        return SyntheticSource()
    return BinanceSource()


def _candles(args: argparse.Namespace, settings: Settings) -> pd.DataFrame:
    limit = args.bars or settings.history_bars
    return _source(args).fetch(settings.symbol, settings.timeframe, limit)


def _percent(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _print_metrics(metrics: dict[str, float]) -> None:
    print(f"  Return          {_percent(metrics['total_return'])}")
    print(f"  Buy and hold    {_percent(metrics['benchmark_return'])}")
    print(f"  Max drawdown    {metrics['max_drawdown'] * 100:.2f}%")
    print(f"  Sharpe ratio    {metrics['sharpe']:.2f}")
    print(f"  Trades          {int(metrics['trades'])}")
    print(f"  Win rate        {metrics['win_rate'] * 100:.1f}%")
    print(f"  Final equity    {metrics['final_equity']:,.2f}")


def cmd_fetch(args: argparse.Namespace, settings: Settings) -> int:
    candles = _candles(args, settings)
    save_csv(candles, args.out)
    print(f"Saved {len(candles)} candles of {settings.symbol} {settings.timeframe} to {args.out}")
    return 0


def cmd_backtest(args: argparse.Namespace, settings: Settings) -> int:
    candles = _candles(args, settings)
    strategy = create_strategy(settings.strategy.name, settings.strategy.params)
    result = run_backtest(candles, strategy, settings.risk, settings.fees, settings.initial_cash)

    print(f"Backtest of {settings.strategy.name} on {settings.symbol} {settings.timeframe}")
    print(f"  Period          {result.timestamps.iloc[0]} to {result.timestamps.iloc[-1]}")
    _print_metrics(result.metrics)
    if result.halted:
        print("  Trading stopped early: the drawdown limit was reached.")
    print()
    print(DISCLAIMER)
    return 0


def cmd_recommend(args: argparse.Namespace, settings: Settings) -> int:
    candles = _candles(args, settings)
    ranked = recommend(
        candles, settings.risk_profile, settings.risk, settings.fees, settings.initial_cash
    )

    print(
        f"Strategies for {settings.symbol} {settings.timeframe}, "
        f"{settings.risk_profile} profile, tested on the newer half of {len(candles)} candles"
    )
    for position, item in enumerate(ranked, start=1):
        label = "suitable" if item.suitable else "not recommended"
        params = ", ".join(f"{key}={value}" for key, value in item.params.items()) or "defaults"
        print()
        print(f"{position}. {item.strategy} ({params}): {label}")
        print(f"  {item.description}")
        if item.metrics:
            _print_metrics(item.metrics)
        for note in item.notes:
            print(f"  - {note}")

    print()
    if not any(item.suitable for item in ranked):
        print("No strategy passed the checks on this data. Staying out is a valid choice.")
    print(DISCLAIMER)
    return 0


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    if settings.mode != "paper":
        print("Live trading is not implemented yet. Set mode: paper in the config.", file=sys.stderr)
        return 2

    trader = Trader(
        settings,
        _source(args),
        create_strategy(settings.strategy.name, settings.strategy.params),
        PaperBroker(settings.initial_cash, settings.fees.fee_rate, settings.fees.slippage_bps),
        RiskManager(settings.risk, peak_equity=settings.initial_cash),
        state_path=settings.state_file,
    )

    if args.once:
        report = trader.step()
        print(
            f"{report.timestamp} price={report.price:.2f} signal={report.signal.name} "
            f"action={report.action} equity={report.equity:,.2f}"
        )
        return 0

    print(f"Paper trading {settings.symbol} {settings.timeframe}. Press Ctrl+C to stop.")
    try:
        trader.run()
    except KeyboardInterrupt:
        print("Stopped.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aitrader", description=__doc__)
    parser.add_argument("--config", help="path to a YAML settings file")
    parser.add_argument("--data", help="read candles from this CSV file")
    parser.add_argument("--synthetic", action="store_true", help="use generated candles")
    parser.add_argument("--bars", type=int, help="how many candles to load")
    parser.add_argument("--verbose", action="store_true", help="log every step")

    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="download candles and save them as CSV")
    fetch.add_argument("--out", required=True, help="CSV file to write")
    fetch.set_defaults(handler=cmd_fetch)

    backtest = commands.add_parser("backtest", help="test the configured strategy on history")
    backtest.set_defaults(handler=cmd_backtest)

    advise = commands.add_parser("recommend", help="rank strategies for the configured profile")
    advise.set_defaults(handler=cmd_recommend)

    run = commands.add_parser("run", help="trade automatically with simulated orders")
    run.add_argument("--once", action="store_true", help="handle one candle and exit")
    run.set_defaults(handler=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        settings = load_settings(args.config)
        return args.handler(args, settings)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
