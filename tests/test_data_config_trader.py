import pytest
from pydantic import ValidationError

from aitrader.bot.trader import Trader
from aitrader.cli import main
from aitrader.config import Settings, load_settings
from aitrader.data.source import CsvSource, parse_klines, save_csv, validate_candles
from aitrader.execution.broker import PaperBroker
from aitrader.risk.manager import RiskManager
from aitrader.strategies import create_strategy


def test_parse_klines():
    rows = [
        [1700003600000, "101.0", "103.0", "100.0", "102.0", "7.5", 1700007199999],
        [1700000000000, "100.0", "102.0", "99.0", "101.0", "5.0", 1700003599999],
    ]
    table = parse_klines(rows)
    assert list(table["close"]) == [101.0, 102.0]  # sorted oldest first
    assert str(table["timestamp"].dt.tz) == "UTC"


def test_validate_candles_reports_missing_columns(candles):
    with pytest.raises(ValueError, match="volume"):
        validate_candles(candles.drop(columns=["volume"]))


def test_csv_round_trip(tmp_path, candles):
    path = tmp_path / "candles.csv"
    save_csv(candles.head(300), path)
    loaded = CsvSource(path).fetch("BTCUSDT", "1h", 100)
    assert len(loaded) == 100
    assert loaded["close"].iloc[-1] == pytest.approx(candles["close"].iloc[299])


def test_default_config_file_loads():
    settings = load_settings("config/default.yaml")
    assert settings.mode == "paper"
    assert settings.strategy.name == "sma_cross"


def test_config_rejects_unknown_keys_and_bad_values():
    with pytest.raises(ValidationError):
        Settings.model_validate({"leverage": 10})
    with pytest.raises(ValidationError):
        Settings.model_validate({"risk": {"risk_per_trade": 0.9}})


class Replay:
    """Feeds the trader one more candle on every fetch."""

    def __init__(self, candles, start):
        self.candles = candles
        self.position = start

    def fetch(self, symbol, timeframe, limit):
        self.position += 1
        return self.candles.iloc[: self.position].tail(limit).reset_index(drop=True)


def make_trader(candles, tmp_path, start=400):
    settings = Settings(strategy={"name": "sma_cross", "params": {"fast": 5, "slow": 15}},
                        history_bars=200)
    return Trader(
        settings,
        Replay(candles, start),
        create_strategy(settings.strategy.name, settings.strategy.params),
        PaperBroker(settings.initial_cash),
        RiskManager(settings.risk, peak_equity=settings.initial_cash),
        state_path=tmp_path / "state.json",
    )


def test_trader_trades_and_keeps_accounts_consistent(candles, tmp_path):
    trader = make_trader(candles, tmp_path)
    actions = [trader.step().action for _ in range(600)]
    assert "buy" in actions
    assert any(action.startswith("sell") for action in actions)
    assert trader.broker.cash >= 0
    assert trader.broker.equity(float(candles["close"].iloc[1000])) > 0


def test_trader_skips_a_candle_it_already_handled(candles, tmp_path):
    trader = make_trader(candles, tmp_path)
    trader.step()
    trader.source.position -= 1
    assert trader.step().action == "wait"


def test_trader_resumes_from_saved_state(candles, tmp_path):
    first = make_trader(candles, tmp_path)
    for _ in range(600):
        first.step()
        if first.broker.quantity > 0:
            break
    assert first.broker.quantity > 0

    second = make_trader(candles, tmp_path, start=first.source.position)
    assert second.broker.quantity == pytest.approx(first.broker.quantity)
    assert second.broker.cash == pytest.approx(first.broker.cash)
    assert second._stop == pytest.approx(first._stop)


def test_cli_backtest_and_recommend_run(capsys):
    assert main(["--synthetic", "--bars", "1500", "backtest"]) == 0
    assert main(["--synthetic", "--bars", "1500", "recommend"]) == 0
    output = capsys.readouterr().out
    assert "Max drawdown" in output
    assert "not financial advice" in output


def test_cli_refuses_live_mode(tmp_path, capsys):
    config = tmp_path / "live.yaml"
    config.write_text("mode: live\n", encoding="utf-8")
    assert main(["--config", str(config), "--synthetic", "run", "--once"]) == 2
    assert "not implemented" in capsys.readouterr().err
