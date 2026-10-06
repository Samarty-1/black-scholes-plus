"""Headless smoke tests of the Streamlit dashboard (no browser, no network)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    assert not at.exception, at.exception
    return at


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_render_has_all_tabs_and_prices(app):
    assert [t.label for t in app.tabs] == [
        "Pricer", "Scenarios", "Strategy builder", "Implied vol", "Smile lab", "Exotics",
        "Convergence"]
    assert float(metric(app, "European (BSM)")) > 0
    # Exotics renders after the Smile lab: guards against an early st.stop().
    assert any(m.label.startswith("down-and-out") for m in app.metric)


def test_spy_snapshot_source_and_heston(app):
    quotes = next(r for r in app.radio if r.label == "Quotes")
    quotes.set_value(quotes.options[1]).run()  # SPY snapshot
    assert not app.exception, app.exception
    assert any("SPY" in md.value and "de-Americanized" in md.value for md in app.markdown)
    app.select_slider[0].set_value(1)
    next(b for b in app.button if b.label == "Calibrate Heston").click().run()
    assert not app.exception, app.exception
    assert any("RMSE" in md.value and "start(s)" in md.value for md in app.markdown)


def test_live_source_waits_for_fetch_without_blocking_other_tabs(app):
    quotes = next(r for r in app.radio if r.label == "Quotes")
    quotes.set_value(quotes.options[2]).run()  # live: nothing fetched yet
    assert not app.exception, app.exception
    assert any("Fetch chain" in i.value for i in app.info)
    assert any(m.label.startswith("down-and-out") for m in app.metric)


def test_cash_dividends_and_bad_input(app):
    app.text_input[0].set_value("30:1.5, 60:1.5, oops").run()
    assert not app.exception, app.exception
    assert float(metric(app, "European (escrowed divs)")) > 0
    assert any("oops" in e.value for e in app.sidebar.error)


def test_put_extremes_and_barrier_monte_carlo(app):
    app.sidebar.radio[0].set_value("put")
    app.sidebar.number_input[2].set_value(1)
    app.sidebar.slider[0].set_value(150.0).run()
    assert not app.exception, app.exception
    next(b for b in app.button if b.label.startswith("Check with Monte Carlo")).click().run()
    assert not app.exception, app.exception
    assert any("z = " in md.value for md in app.markdown)
