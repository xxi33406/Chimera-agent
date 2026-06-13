"""Tests for billing/wallet/tokenjuice cost tracking system."""

from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path):
    """Create a fresh SessionDB rooted at the pytest tmp_path."""
    from hermes_state import SessionDB

    return SessionDB(db_path=tmp_path / "state.db")


# ---------------------------------------------------------------------------
# Wallet data layer
# ---------------------------------------------------------------------------


class TestWalletDataLayer:
    """Test wallet table operations in SessionDB."""

    def test_get_wallet_creates_default(self, tmp_path):
        """get_wallet() auto-creates default wallet row."""
        db = _make_db(tmp_path)
        wallet = db.get_wallet()
        assert wallet["id"] == "default"
        assert wallet["spent_today_usd"] == 0.0
        assert wallet["spent_month_usd"] == 0.0
        assert wallet["spent_total_usd"] == 0.0
        assert wallet["warning_sent_today"] == 0

    def test_charge_wallet_increments(self, tmp_path):
        """charge_wallet() adds to all counters."""
        db = _make_db(tmp_path)
        wallet = db.charge_wallet(0.50, source="main_model")
        assert wallet["spent_today_usd"] == pytest.approx(0.50)
        assert wallet["spent_month_usd"] == pytest.approx(0.50)
        assert wallet["spent_total_usd"] == pytest.approx(0.50)

        wallet = db.charge_wallet(0.25, source="auxiliary_dream")
        assert wallet["spent_today_usd"] == pytest.approx(0.75)
        assert wallet["spent_month_usd"] == pytest.approx(0.75)
        assert wallet["spent_total_usd"] == pytest.approx(0.75)

    def test_charge_wallet_zero_noop(self, tmp_path):
        """charge_wallet(0) or negative does not change state."""
        db = _make_db(tmp_path)
        db.charge_wallet(1.0, source="test")
        wallet = db.charge_wallet(0.0, source="test")
        assert wallet["spent_today_usd"] == pytest.approx(1.0)
        wallet = db.charge_wallet(-0.5, source="test")
        assert wallet["spent_today_usd"] == pytest.approx(1.0)

    def test_daily_auto_reset(self, tmp_path):
        """get_wallet() resets daily counter when day changes."""
        db = _make_db(tmp_path)
        db.charge_wallet(2.0, source="test")

        # Simulate yesterday's reset timestamp via the public write helper
        # so the underlying transaction machinery is exercised.
        yesterday = time.time() - 86400 * 2

        def _set_reset(conn):
            conn.execute(
                "UPDATE wallet SET day_reset_at = ? WHERE id = 'default'",
                (yesterday,),
            )

        db._execute_write(_set_reset)

        wallet = db.get_wallet()
        assert wallet["spent_today_usd"] == 0.0
        # Monthly and total should remain
        assert wallet["spent_month_usd"] == pytest.approx(2.0)
        assert wallet["spent_total_usd"] == pytest.approx(2.0)

    def test_monthly_auto_reset(self, tmp_path):
        """get_wallet() resets monthly counter when month changes."""
        db = _make_db(tmp_path)
        db.charge_wallet(10.0, source="test")

        # Simulate last month's reset timestamp.  We push BOTH the day
        # reset and month reset back into a previous month so the daily
        # auto-reset (which fires first) leaves spent_today_usd at 0 too.
        last_month = time.time() - 86400 * 35

        def _set_reset(conn):
            conn.execute(
                "UPDATE wallet SET month_reset_at = ?, day_reset_at = ? "
                "WHERE id = 'default'",
                (last_month, last_month),
            )

        db._execute_write(_set_reset)

        wallet = db.get_wallet()
        assert wallet["spent_month_usd"] == 0.0
        assert wallet["spent_total_usd"] == pytest.approx(10.0)

    def test_reset_wallet_daily(self, tmp_path):
        """Manual daily reset works."""
        db = _make_db(tmp_path)
        db.charge_wallet(3.0, source="test")
        db.reset_wallet_daily()
        wallet = db.get_wallet()
        assert wallet["spent_today_usd"] == 0.0
        assert wallet["warning_sent_today"] == 0

    def test_reset_wallet_monthly(self, tmp_path):
        """Manual monthly reset works."""
        db = _make_db(tmp_path)
        db.charge_wallet(20.0, source="test")
        db.reset_wallet_monthly()
        wallet = db.get_wallet()
        assert wallet["spent_month_usd"] == 0.0

    def test_mark_warning_sent(self, tmp_path):
        """_mark_warning_sent sets the per-day flag."""
        db = _make_db(tmp_path)
        db.get_wallet()  # ensure exists
        db._mark_warning_sent()
        wallet = db.get_wallet()
        assert wallet["warning_sent_today"] == 1


# ---------------------------------------------------------------------------
# Budget enforcement (_check_billing_budget)
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    """Test _check_billing_budget logic."""

    def _make_mock_agent(
        self,
        daily_limit: float = 5.0,
        monthly_limit: float = 50.0,
        threshold: float = 0.8,
        hard_stop: bool = False,
    ):
        """Create a mock agent with billing config."""
        agent = MagicMock()
        agent._billing_enabled = True
        agent._billing_config = {
            "enabled": True,
            "daily_limit_usd": daily_limit,
            "monthly_limit_usd": monthly_limit,
            "warning_threshold": threshold,
            "hard_stop": hard_stop,
            "track_auxiliary": True,
        }
        return agent

    def test_no_action_zero_charge(self):
        """No budget check for zero amount."""
        from agent.conversation_loop import _check_billing_budget

        agent = self._make_mock_agent()
        agent._session_db = MagicMock()
        _check_billing_budget(agent, 0.0)
        agent._session_db.charge_wallet.assert_not_called()

    def test_no_action_when_no_session_db(self):
        """No-op when agent has no session DB attached."""
        from agent.conversation_loop import _check_billing_budget

        agent = self._make_mock_agent()
        agent._session_db = None
        # Should not raise
        _check_billing_budget(agent, 1.0)

    def test_warning_at_threshold(self):
        """Warning logged when daily spend reaches threshold."""
        from agent.conversation_loop import _check_billing_budget

        agent = self._make_mock_agent(daily_limit=10.0, threshold=0.8)

        mock_db = MagicMock()
        mock_db.charge_wallet.return_value = {
            "spent_today_usd": 8.5,  # 85% > 80% threshold
            "spent_month_usd": 8.5,
            "warning_sent_today": 0,
        }
        agent._session_db = mock_db

        _check_billing_budget(agent, 0.5)

        mock_db._mark_warning_sent.assert_called_once()

    def test_hard_stop_daily(self):
        """BudgetExceededError raised when hard_stop and daily exceeded."""
        from agent.conversation_loop import (
            BudgetExceededError,
            _check_billing_budget,
        )

        agent = self._make_mock_agent(daily_limit=5.0, hard_stop=True)

        mock_db = MagicMock()
        mock_db.charge_wallet.return_value = {
            "spent_today_usd": 5.5,  # Over limit
            "spent_month_usd": 5.5,
            "warning_sent_today": 0,
        }
        agent._session_db = mock_db

        with pytest.raises(BudgetExceededError, match="Daily budget"):
            _check_billing_budget(agent, 0.5)

    def test_hard_stop_monthly(self):
        """BudgetExceededError raised when hard_stop and monthly exceeded."""
        from agent.conversation_loop import (
            BudgetExceededError,
            _check_billing_budget,
        )

        # daily_limit=0 disables the daily branch, so the monthly check fires.
        agent = self._make_mock_agent(
            monthly_limit=50.0, daily_limit=0, hard_stop=True
        )

        mock_db = MagicMock()
        mock_db.charge_wallet.return_value = {
            "spent_today_usd": 5.0,
            "spent_month_usd": 55.0,  # Over monthly limit
            "warning_sent_today": 0,
        }
        agent._session_db = mock_db

        with pytest.raises(BudgetExceededError, match="Monthly budget"):
            _check_billing_budget(agent, 1.0)

    def test_no_stop_without_hard_stop(self):
        """No error when over limit but hard_stop=False."""
        from agent.conversation_loop import _check_billing_budget

        agent = self._make_mock_agent(daily_limit=5.0, hard_stop=False)

        mock_db = MagicMock()
        mock_db.charge_wallet.return_value = {
            "spent_today_usd": 10.0,  # Way over limit
            "spent_month_usd": 10.0,
            "warning_sent_today": 1,  # Warning already sent
        }
        agent._session_db = mock_db

        # Should not raise
        _check_billing_budget(agent, 1.0)


# ---------------------------------------------------------------------------
# Auxiliary cost tracking
# ---------------------------------------------------------------------------


class TestAuxiliaryCostTracking:
    """Test _track_auxiliary_cost function."""

    def test_tracks_positive_cost(self):
        """Positive cost gets charged to wallet."""
        from agent.auxiliary_client import _track_auxiliary_cost

        mock_db = MagicMock()
        mock_db.charge_wallet.return_value = {"spent_today_usd": 0.01}

        # estimate_usage_cost is imported lazily inside the function from
        # agent.usage_pricing, so we patch the source module symbol.
        fake_result = MagicMock()
        fake_result.amount_usd = Decimal("0.005")

        with patch(
            "agent.auxiliary_client._get_wallet_db", return_value=mock_db
        ), patch(
            "agent.usage_pricing.estimate_usage_cost", return_value=fake_result
        ):
            _track_auxiliary_cost(
                model="claude-3-haiku-20240307",
                input_tokens=1000,
                output_tokens=200,
                task="dream",
            )

        mock_db.charge_wallet.assert_called_once()
        call_args = mock_db.charge_wallet.call_args
        # First positional argument is the dollar amount.
        assert call_args.args[0] == pytest.approx(0.005)
        # Source kwarg should encode the task name.
        assert call_args.kwargs.get("source") == "auxiliary_dream"

    def test_zero_cost_no_charge(self):
        """Zero/None amount_usd does not trigger a wallet charge."""
        from agent.auxiliary_client import _track_auxiliary_cost

        mock_db = MagicMock()
        fake_result = MagicMock()
        fake_result.amount_usd = Decimal("0")

        with patch(
            "agent.auxiliary_client._get_wallet_db", return_value=mock_db
        ), patch(
            "agent.usage_pricing.estimate_usage_cost", return_value=fake_result
        ):
            _track_auxiliary_cost(
                model="some-model",
                input_tokens=10,
                output_tokens=5,
                task="unit",
            )

        mock_db.charge_wallet.assert_not_called()

    def test_never_crashes(self):
        """Tracking errors are swallowed silently."""
        from agent.auxiliary_client import _track_auxiliary_cost

        with patch(
            "agent.auxiliary_client._get_wallet_db",
            side_effect=Exception("DB error"),
        ):
            # Should not raise
            _track_auxiliary_cost(
                model="nonexistent-model",
                input_tokens=100,
                output_tokens=50,
                task="test",
            )


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------


class TestBillingConfig:
    """Test billing configuration defaults."""

    def test_default_config_has_billing(self):
        """DEFAULT_CONFIG includes billing section with expected fields."""
        from hermes_cli.config import DEFAULT_CONFIG

        assert "billing" in DEFAULT_CONFIG
        billing = DEFAULT_CONFIG["billing"]
        assert billing["enabled"] is True
        assert billing["daily_limit_usd"] == 5.0
        assert billing["monthly_limit_usd"] == 50.0
        assert billing["warning_threshold"] == 0.8
        assert billing["hard_stop"] is False
        assert billing["track_auxiliary"] is True


# ---------------------------------------------------------------------------
# CLI presentation helpers
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Test gauge bar and token formatting helpers."""

    def test_make_gauge_bar_empty(self):
        """0% ratio produces all empty blocks."""
        from cli import _make_gauge_bar

        bar = _make_gauge_bar(0.0, 10)
        assert bar.count("█") == 0
        assert "░" in bar

    def test_make_gauge_bar_full(self):
        """100% ratio produces all filled blocks."""
        from cli import _make_gauge_bar

        bar = _make_gauge_bar(1.0, 10)
        assert "█" in bar
        assert bar.count("█") == 10

    def test_make_gauge_bar_color_green(self):
        """Low ratio uses green color."""
        from cli import _make_gauge_bar

        bar = _make_gauge_bar(0.3, 10)
        assert "green" in bar

    def test_make_gauge_bar_color_yellow(self):
        """Medium ratio uses yellow color."""
        from cli import _make_gauge_bar

        bar = _make_gauge_bar(0.75, 10)
        assert "yellow" in bar

    def test_make_gauge_bar_color_red(self):
        """High ratio uses red color."""
        from cli import _make_gauge_bar

        bar = _make_gauge_bar(0.95, 10)
        assert "red" in bar

    def test_make_gauge_bar_clamps(self):
        """Out-of-range ratios are clamped to [0, 1]."""
        from cli import _make_gauge_bar

        # Negative ratio behaves like 0%
        bar_low = _make_gauge_bar(-0.5, 10)
        assert bar_low.count("█") == 0
        # >100% behaves like 100%
        bar_high = _make_gauge_bar(2.0, 10)
        assert bar_high.count("█") == 10

    def test_format_tokens_small(self):
        """Small numbers shown as-is."""
        from cli import _format_tokens

        assert _format_tokens(500) == "500"

    def test_format_tokens_thousands(self):
        """Thousands shown as K."""
        from cli import _format_tokens

        assert _format_tokens(125000) == "125K"

    def test_format_tokens_millions(self):
        """Millions shown as M."""
        from cli import _format_tokens

        assert _format_tokens(1_500_000) == "1.5M"
