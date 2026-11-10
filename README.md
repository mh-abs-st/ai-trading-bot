# AI Trading Bot

A cryptocurrency trading bot that learns from price history, recommends strategies that fit your risk settings, and trades automatically.

**Status: early version, paper trading only.** Orders are simulated against real or recorded prices. Placing real orders on an exchange is not implemented yet.

> Trading cryptocurrencies can lose money quickly. Nothing this software prints is financial advice, and a good backtest does not predict future results.

## What works today

- **Market data** from Binance's public API (no account needed), from CSV files, or generated for demos
- **Three strategies**: moving-average crossover, RSI mean reversion, and a machine-learning model that estimates the chance of a rise
- **Backtesting** with fees, slippage, stop-loss, take-profit and a drawdown stop
- **Strategy recommendations** ranked for a conservative, balanced or aggressive risk profile
- **Automated paper trading** that follows your settings and survives restarts

## Quick start

Requires Python 3.11 or newer.

```
pip install -e ".[dev]"
pytest
```

Try it without any network access, on generated data:

```
aitrader --synthetic --bars 3000 recommend
aitrader --synthetic --bars 3000 backtest
```

With real market data:

```
aitrader --config config/default.yaml fetch --out data/btc_1h.csv --bars 5000
aitrader --config config/default.yaml --data data/btc_1h.csv recommend
aitrader --config config/default.yaml --data data/btc_1h.csv backtest
aitrader --config config/default.yaml run
```

`run` checks for a new candle every `poll_seconds`, decides, and places a simulated order. Its balances and open position are kept in `state_file`, so stopping and starting it again continues where it left off.

## Settings

Everything is set in one YAML file; see [`config/default.yaml`](config/default.yaml).

| Setting | Meaning |
| --- | --- |
| `symbol`, `timeframe` | Market and candle size, for example `BTCUSDT` and `1h` |
| `strategy.name`, `strategy.params` | Which strategy trades, and its parameters |
| `risk_profile` | How recommendations weigh drawdown against return |
| `risk.risk_per_trade` | Share of equity lost if a trade hits its stop |
| `risk.stop_loss_pct`, `risk.take_profit_pct` | Exit levels relative to the entry price |
| `risk.max_position_fraction` | Largest share of equity in one position |
| `risk.max_drawdown_pct` | Trading halts when equity falls this far from its peak |
| `fees` | Fee rate and slippage used in simulations |

Exchange keys, once live trading exists, are read from environment variables only; see `.env.example`.

## How it is organised

See [`docs/architecture.md`](docs/architecture.md) for the design and how to add a strategy.

| Package | Role |
| --- | --- |
| `aitrader.data` | Candle sources |
| `aitrader.features` | Indicators and the ML feature table |
| `aitrader.strategies` | Strategies and their registry |
| `aitrader.risk` | Position sizing and trading halts |
| `aitrader.execution` | Brokers: simulated today, live later |
| `aitrader.backtest` | Historical simulation and metrics |
| `aitrader.recommend` | Strategy ranking per risk profile |
| `aitrader.bot` | The automated trading loop |
