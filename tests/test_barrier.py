import itertools

import pytest

import bsplus as bs
from bsplus.barrier import KINDS, barrier_mc, barrier_price

# Haug (2007), Table 4-13: S=100, T=0.5, r=0.08, b=0.04 (q=0.04), sigma=0.25,
# cash rebate 3.
HAUG = [
    ("down-and-out", "call", 90, 95, 9.0246),
    ("down-and-in", "call", 90, 95, 7.7627),
    ("up-and-out", "call", 90, 105, 2.6789),
    ("up-and-in", "call", 90, 105, 14.1112),
    ("down-and-out", "put", 90, 95, 2.2798),
    ("down-and-in", "put", 90, 95, 2.9586),
    ("up-and-out", "put", 90, 105, 3.7760),
    ("up-and-in", "put", 90, 105, 1.4653),
]


@pytest.mark.parametrize("kind,typ,K,H,expected", HAUG)
def test_haug_reference_values(kind, typ, K, H, expected):
    value = barrier_price(100, K, H, 0.5, 0.08, 0.25, typ, kind, q=0.04, rebate=3.0)
    assert value == pytest.approx(expected, abs=1e-4)


CASES = [
    (typ, K, H, kind)
    for typ, K, H in itertools.product(["call", "put"], [90, 110], [95, 105])
    for kind in KINDS
    if (kind.startswith("down") and H < 100) or (kind.startswith("up") and H > 100)
]


@pytest.mark.parametrize("typ,K,H,kind", [c for c in CASES if c[3].endswith("-in")])
def test_in_plus_out_equals_vanilla(typ, K, H, kind):
    out = kind.replace("-in", "-out")
    total = (barrier_price(100, K, H, 0.5, 0.08, 0.25, typ, kind, q=0.04)
             + barrier_price(100, K, H, 0.5, 0.08, 0.25, typ, out, q=0.04))
    assert total == pytest.approx(bs.price(100, K, 0.5, 0.08, 0.25, typ, 0.04), abs=1e-10)


@pytest.mark.parametrize("typ,K,H,kind", CASES)
def test_closed_form_matches_brownian_bridge_monte_carlo(typ, K, H, kind):
    cf = barrier_price(100, K, H, 0.5, 0.08, 0.25, typ, kind, q=0.04)
    mc, se = barrier_mc(100, K, H, 0.5, 0.08, 0.25, typ, kind, q=0.04,
                        n_paths=60_000, n_steps=60, seed=11)
    assert abs(mc - cf) <= 4 * se + 1e-12


def test_already_breached_barrier():
    assert barrier_price(90, 100, 95, 1, 0.05, 0.2, "call", "down-and-out", rebate=2) == 2
    vanilla = bs.price(90, 100, 1, 0.05, 0.2, "call")
    assert barrier_price(90, 100, 95, 1, 0.05, 0.2, "call", "down-and-in") == pytest.approx(vanilla)


def test_invalid_kind():
    with pytest.raises(ValueError):
        barrier_price(100, 100, 90, 1, 0.05, 0.2, "call", "sideways-and-out")
