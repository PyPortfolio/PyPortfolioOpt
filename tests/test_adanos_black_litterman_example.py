from copy import deepcopy
import doctest
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RETURNS = [0.049440968872525, 0.054098781465159, 0.049337572652089]
EXPECTED_COVARIANCE = [
    [0.072115611553356, 0.063060104777357, 0.048733548111661],
    [0.063060104777357, 0.096036003132994, 0.063313193334073],
    [0.048733548111661, 0.063313193334073, 0.109154131959175],
]


@pytest.fixture
def example(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("The saved-research example must stay offline")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.chdir(ROOT)
    path = ROOT / "docs" / "BlackLitterman.rst"
    test = doctest.DocTestParser().get_doctest(
        path.read_text(encoding="utf-8"), {}, "adanos-black-litterman", str(path), 0
    )
    assert test.examples, "Missing executable Adanos Black-Litterman recipe"
    doctest.DebugRunner().run(test, clear_globs=False)
    return test.globs


def run_example(example, **changes):
    inputs = {
        name: deepcopy(example[name])
        for name in ("prices", "saved_reports", "views", "confidences", "observed_at")
    }
    inputs.update(changes)
    return example["research_example"](**inputs)


def test_documented_native_posterior_and_audit(example):
    audit, model = run_example(example)
    assert audit.index.tolist() == ["GOOG", "AAPL"]
    np.testing.assert_allclose(audit["annual_view"], [0.06, 0.05])
    np.testing.assert_allclose(audit["confidence"], [0.7, 0.4])
    np.testing.assert_allclose(audit["sentiment_score"], [-0.2, 0.3])
    assert audit["mentions"].tolist() == [8, 12]
    assert pd.isna(audit.loc["GOOG", "buzz_score"])
    assert audit["source"].tolist() == [
        "https://api.adanos.com/reddit/stocks/v1/stock/GOOG",
        "https://api.adanos.com/reddit/stocks/v1/stock/AAPL",
    ]
    assert audit["date"].eq(pd.Timestamp("2018-04-11")).all()
    assert audit["observed_at"].eq(pd.Timestamp("2018-04-12T12:00:00Z")).all()
    np.testing.assert_allclose(model.pi.ravel(), [0.04, 0.04, 0.04])
    posterior = model.bl_returns()
    assert posterior.index.tolist() == ["AAPL", "GOOG", "AMZN"]
    np.testing.assert_allclose(posterior, EXPECTED_RETURNS, rtol=1e-7)
    covariance = model.bl_cov()
    assert covariance.index.equals(posterior.index)
    assert covariance.columns.equals(posterior.index)
    np.testing.assert_allclose(covariance, EXPECTED_COVARIANCE, rtol=1e-7)


def test_research_is_context_not_a_return_or_confidence(example):
    audit, original = run_example(example)
    reports = deepcopy(example["saved_reports"])[::-1]
    for report in reports:
        report["daily_trend"][0].update(
            sentiment_score=0.95, buzz_score=99.0, mentions=999
        )
    changed_audit, fixed = run_example(example, saved_reports=reports)
    assert not audit["sentiment_score"].equals(changed_audit["sentiment_score"])
    pd.testing.assert_series_equal(original.bl_returns(), fixed.bl_returns())
    pd.testing.assert_frame_equal(original.bl_cov(), fixed.bl_cov())
    views = example["views"].copy()
    views.loc["GOOG"] = 0.12
    _, changed_view = run_example(example, views=views)
    assert changed_view.bl_returns()["GOOG"] > original.bl_returns()["GOOG"]
    confidences = example["confidences"].copy()
    confidences.loc["GOOG"] = 0.2
    _, changed_confidence = run_example(example, confidences=confidences)
    assert changed_confidence.bl_returns()["GOOG"] < original.bl_returns()["GOOG"]


def test_confidences_follow_view_order_not_response_order(example):
    _, original = run_example(example)
    audit, reordered = run_example(
        example,
        views=example["views"].iloc[::-1],
        confidences=example["confidences"].iloc[::-1],
        saved_reports=example["saved_reports"][::-1],
    )
    assert audit.index.tolist() == ["AAPL", "GOOG"]
    np.testing.assert_allclose(audit["confidence"], [0.4, 0.7])
    np.testing.assert_allclose(original.bl_returns(), reordered.bl_returns())


@pytest.mark.parametrize("missing", ["report", "daily_trend", "found", "error"])
def test_missing_context_is_explicit(example, missing):
    reports = deepcopy(example["saved_reports"])
    if missing == "report":
        reports.pop()
    elif missing == "daily_trend":
        reports[0]["daily_trend"] = None
    elif missing == "found":
        reports[0]["found"] = False
    else:
        reports[0] = {"detail": "Historical data limit exceeded"}
    with pytest.raises(ValueError):
        run_example(example, saved_reports=reports)


def test_null_metrics_do_not_become_neutral_or_add_views(example):
    reports = deepcopy(example["saved_reports"])
    reports[0]["daily_trend"][0]["sentiment_score"] = None
    audit, model = run_example(example, saved_reports=reports)
    assert pd.isna(audit.loc["AAPL", "sentiment_score"])
    assert "AMZN" not in audit.index
    np.testing.assert_allclose(model.bl_returns(), EXPECTED_RETURNS, rtol=1e-7)


@pytest.mark.parametrize("problem", ["missing", "bounds", "unknown", "duplicate"])
def test_manual_input_errors(example, problem):
    views = example["views"].copy()
    confidences = example["confidences"].copy()
    if problem == "missing":
        confidences = confidences.drop("GOOG")
    elif problem == "bounds":
        confidences.loc["GOOG"] = 1.1
    elif problem == "unknown":
        views = views.rename(index={"GOOG": "UNKNOWN"})
        confidences = confidences.rename(index={"GOOG": "UNKNOWN"})
    else:
        views.index = ["AAPL", "AAPL"]
    with pytest.raises(ValueError):
        run_example(example, views=views, confidences=confidences)


@pytest.mark.parametrize(
    "confidences",
    [
        pytest.param(pd.Series({"AAPL": 0.4}, dtype="Float64"), id="nullable-missing"),
        pytest.param(
            pd.Series({"AAPL": 0.4, "GOOG": pd.NA}, dtype="Float64"), id="nullable-na"
        ),
        pytest.param(pd.Series({"AAPL": 0.4, "GOOG": np.nan}), id="nan"),
        pytest.param(pd.Series({"AAPL": 0.4, "GOOG": np.inf}), id="positive-infinity"),
        pytest.param(pd.Series({"AAPL": 0.4, "GOOG": -np.inf}), id="negative-infinity"),
    ],
)
def test_missing_or_nonfinite_confidences_are_rejected(example, confidences):
    with pytest.raises(ValueError, match="Supply a confidence"):
        run_example(example, confidences=confidences)


@pytest.mark.parametrize("problem", ["duplicate", "date", "nonfinite", "range"])
def test_invalid_saved_evidence_is_rejected(example, problem):
    reports = deepcopy(example["saved_reports"])
    if problem == "duplicate":
        reports.append(deepcopy(reports[0]))
    elif problem == "date":
        reports[0]["daily_trend"][0]["date"] = "not-a-date"
    elif problem == "nonfinite":
        reports[0]["daily_trend"][0]["sentiment_score"] = float("inf")
    else:
        reports[0]["daily_trend"][0]["buzz_score"] = 101
    with pytest.raises(ValueError):
        run_example(example, saved_reports=reports)
