import numpy as np
import pytest

import bsplus as bs
from bsplus.american import american_price, european_tree_price


@pytest.mark.parametrize("S,expected", [(90, 0.0205), (100, 1.8757), (110, 10.0)])
def test_haug_american_call_on_futures_like_asset(S, expected):
    # Haug (2007) Table: American call, K=100, T=0.1, r=0.10, b=0 (q=r), sigma=0.15.
    # Reference values are high-resolution binomial prices.
    value = american_price(S, 100, 0.1, 0.10, 0.15, "call", q=0.10, steps=1001).price
    assert value == pytest.approx(expected, abs=0.015)


def test_leisen_reimer_converges_to_black_scholes():
    exact = bs.price(100, 105, 1.0, 0.05, 0.25, "put")
    lr = european_tree_price(100, 105, 1.0, 0.05, 0.25, "put", steps=101, method="lr")
    crr = european_tree_price(100, 105, 1.0, 0.05, 0.25, "put", steps=101, method="crr")
    assert abs(lr - exact) < 1e-3
    assert abs(lr - exact) < abs(crr - exact)  # LR beats CRR at equal steps


def test_lr_and_crr_agree_on_american_put():
    lr = american_price(100, 100, 1.0, 0.05, 0.2, "put", steps=501, method="lr").price
    crr = american_price(100, 100, 1.0, 0.05, 0.2, "put", steps=4001, method="crr").price
    assert lr == pytest.approx(crr, abs=2e-3)


def test_american_call_without_dividends_equals_european():
    # Never optimal to exercise a call early when q = 0 (Merton 1973).
    res = american_price(100, 95, 1.0, 0.05, 0.3, "call", q=0.0)
    assert res.price == pytest.approx(bs.price(100, 95, 1.0, 0.05, 0.3, "call"), abs=1e-3)
    assert res.early_exercise_premium < 1e-3


def test_american_put_has_positive_early_exercise_premium():
    res = american_price(100, 110, 1.0, 0.08, 0.2, "put")
    assert res.early_exercise_premium > 0.1
    assert res.price >= 110 - 100  # never below immediate exercise value


def test_tree_greeks_close_to_black_scholes_for_european_like_case():
    res = american_price(100, 100, 0.5, 0.0, 0.25, "call", q=0.0, steps=501)
    g = bs.greeks(100, 100, 0.5, 0.0, 0.25, "call")
    assert res.delta == pytest.approx(g.delta, abs=2e-3)
    assert res.gamma == pytest.approx(g.gamma, rel=0.02)
    assert res.theta == pytest.approx(g.theta, rel=0.02)


def test_deep_itm_put_is_exercised():
    res = american_price(50, 100, 1.0, 0.05, 0.2, "put")
    assert res.price == pytest.approx(50.0, abs=1e-6)
    assert res.delta == pytest.approx(-1.0, abs=1e-6)


def test_invalid_tree_inputs():
    with pytest.raises(ValueError):
        american_price(100, 100, 0.0, 0.05, 0.2)
    with pytest.raises(ValueError):
        american_price(100, 100, 1.0, 0.05, 0.2, method="trinomial")
    assert np.isfinite(american_price(100, 100, 1.0, 0.05, 0.2, steps=200).price)
