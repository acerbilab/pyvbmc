"""Lightweight checks for the stored final-boost comparison helper."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "dev" / "scripts" / "boost_comparison.py"
pytestmark = pytest.mark.skipif(
    not SCRIPT.is_file(),
    reason="developer scripts require a repository checkout",
)


def _load_script():
    scripts = str(SCRIPT.parent)
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location(
            "boost_comparison", SCRIPT
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts)


@pytest.fixture(scope="module")
def comparison():
    return _load_script()


def _side(final_elbo, final_sd):
    return {
        "label": "normal_D5",
        "seed": 3,
        "final": {
            "best_iter": 1,
            "elbo": final_elbo,
            "elbo_sd": final_sd,
        },
    }


def test_score_record_checks_both_endpoints_and_strict_boundary(comparison):
    trace = {"elbo": np.array([-2.0, -1.0]), "elbo_sd": np.array([0.2, 0.1])}
    row = comparison.score_record(_side(-0.9, 0.16), trace)

    assert row["endpoint_b0"] == pytest.approx(0.1)
    assert row["endpoint_b5"] == pytest.approx(-0.2)
    assert row["worst_b"] == 5
    assert row["rejected"] == {"0.1": True, "0.2": True}


def test_score_record_marks_invalid_uncertainty(comparison):
    trace = {"elbo": np.array([0.0, 1.0]), "elbo_sd": np.array([0.1, -0.1])}
    row = comparison.score_record(_side(1.1, 0.1), trace)

    assert not row["valid"]
    assert row["score_drop"] is None
    assert row["rejected"] == {"0.1": None, "0.2": None}


def test_build_vp_recovers_ragged_iteration_and_final_arrays(comparison):
    trace = {
        "vp_iter": np.array([0, 1, 1]),
        "vp_w": np.array([1.0, 0.25, 0.75]),
        "vp_mu": np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]),
        "vp_sigma": np.array([1.0, 0.5, 0.6]),
        "vp_lambd": np.array([[1.0, 1.0], [2.0, 3.0]]),
        "final_w": np.array([1.0]),
        "final_mu": np.array([[5.0], [6.0]]),
        "final_sigma": np.array([0.7]),
        "final_lambd": np.array([4.0, 5.0]),
    }
    pt = comparison.ParameterTransformer(2)

    before = comparison._build_vp(trace, pt, iteration=1)
    after = comparison._build_vp(trace, pt)

    assert before.K == 2
    assert np.array_equal(before.mu, np.array([[1.0, 3.0], [2.0, 4.0]]))
    assert np.array_equal(before.lambd, np.array([[2.0], [3.0]]))
    assert after.K == 1
    assert np.array_equal(after.mu, np.array([[5.0], [6.0]]))
