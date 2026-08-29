import math
from loralink_reviewer_response.statlib import mean_std_ci

def test_single_value_has_degenerate_ci():
    r = mean_std_ci([2.0])
    assert r["n"] == 1 and r["mean"] == 2.0
    assert r["ci_lo"] == 2.0 and r["ci_hi"] == 2.0 and r["std"] == 0.0

def test_known_sample():
    r = mean_std_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert math.isclose(r["mean"], 3.0)
    assert math.isclose(r["std"], math.sqrt(2.5))          # ddof=1
    assert r["ci_lo"] < 3.0 < r["ci_hi"]
    assert math.isclose(r["ci_hi"] - 3.0, 3.0 - r["ci_lo"], rel_tol=1e-9)
