import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pypfopt import expected_returns
from tests.utilities_for_tests import get_data

mvsk_installed = _check_soft_dependencies("yand-mvsk", severity="none")


def setup_efficient_mvsk(**kwargs):
    from pypfopt import EfficientMVSK

    df = get_data().dropna(axis=0, how="any")
    historic_returns = expected_returns.returns_from_prices(df)
    return EfficientMVSK(historic_returns, **kwargs)


def setup_efficient_mvsk_from_prices(**kwargs):
    from pypfopt import EfficientMVSK

    df = get_data().dropna(axis=0, how="any")
    return EfficientMVSK.from_prices(df, **kwargs)


@pytest.mark.skipif(not mvsk_installed, reason="yand-mvsk not installed")
class TestEfficientMVSK:
    def test_mvsk_example(self):
        mv = setup_efficient_mvsk()
        w = mv.min_mvsk()

        assert isinstance(w, dict)
        assert set(w.keys()) == set(mv.tickers)
        np.testing.assert_almost_equal(mv.weights.sum(), 1)
        assert all(i >= -1e-5 for i in w.values())

    def test_mvsk_converged(self):
        mv = setup_efficient_mvsk()
        mv.min_mvsk()
        assert mv._result.converged

    def test_mvsk_beats_equal_weight(self):
        mv = setup_efficient_mvsk()
        mv.min_mvsk()
        from yand_mvsk import MVSKOracle

        oracle = MVSKOracle(mv.returns, mv.c)
        eq = np.ones(mv.n_assets) / mv.n_assets
        assert oracle.value(mv.weights) <= oracle.value(eq)

    def test_mvsk_weights_long_only(self):
        mv = setup_efficient_mvsk()
        mv.min_mvsk()
        assert np.all(mv.weights >= -1e-8)

    def test_mvsk_clean_weights(self):
        mv = setup_efficient_mvsk()
        mv.min_mvsk()
        clean = mv.clean_weights(cutoff=1e-4, rounding=4)
        assert isinstance(clean, dict)
        assert set(clean.keys()) == set(mv.tickers)
        assert all(v >= 0 for v in clean.values())

    def test_mvsk_portfolio_performance(self):
        mv = setup_efficient_mvsk()
        mv.min_mvsk()
        perf = mv.portfolio_performance()

        assert isinstance(perf, tuple)
        assert len(perf) == 5
        mu, vol, sharpe, skew, kurt = perf
        assert isinstance(mu, float)
        assert vol > 0

    def test_mvsk_portfolio_performance_verbose(self, capsys):
        mv = setup_efficient_mvsk()
        mv.min_mvsk()
        perf = mv.portfolio_performance(verbose=True)
        captured = capsys.readouterr()
        assert "Expected annual return" in captured.out
        assert "Sharpe Ratio" in captured.out
        assert "Skewness" in captured.out

        perf2 = mv.portfolio_performance()
        np.testing.assert_equal(perf, perf2)

    def test_mvsk_performance_before_optimize_raises(self):
        mv = setup_efficient_mvsk()
        with pytest.raises(ValueError):
            mv.portfolio_performance()

    def test_mvsk_from_prices(self):
        mv = setup_efficient_mvsk_from_prices()
        w = mv.min_mvsk()
        assert isinstance(w, dict)
        np.testing.assert_almost_equal(mv.weights.sum(), 1)

    def test_mvsk_from_prices_numpy(self):
        from pypfopt import EfficientMVSK

        df = get_data().dropna(axis=0, how="any")
        mv = EfficientMVSK.from_prices(df.values)
        w = mv.min_mvsk()
        assert isinstance(w, dict)
        np.testing.assert_almost_equal(mv.weights.sum(), 1)

    def test_mvsk_custom_gamma(self):
        mv2 = setup_efficient_mvsk(gamma=2.0)
        mv2.min_mvsk()

        mv20 = setup_efficient_mvsk(gamma=20.0)
        mv20.min_mvsk()

        vol2 = mv2.portfolio_performance()[1]
        vol20 = mv20.portfolio_performance()[1]
        assert vol2 >= vol20

    def test_mvsk_custom_coefficients(self):
        from yand_mvsk import crra_coefficients

        c = crra_coefficients(6.0)
        mv = setup_efficient_mvsk(c=c)
        w = mv.min_mvsk()
        assert isinstance(w, dict)
        np.testing.assert_almost_equal(mv.weights.sum(), 1)

    def test_mvsk_convex_property(self):
        mv = setup_efficient_mvsk(gamma=6.0)
        assert mv.convex

    def test_mvsk_weight_bounds_lower(self):
        mv = setup_efficient_mvsk(weight_bounds=(0.01, 1))
        mv.min_mvsk()
        assert np.all(mv.weights >= 0.01 - 1e-6)

    def test_mvsk_upper_bounds_raises(self):
        with pytest.raises(NotImplementedError):
            setup_efficient_mvsk(weight_bounds=(0, 0.5))

    def test_mvsk_bad_bounds_raises(self):
        with pytest.raises(TypeError):
            setup_efficient_mvsk(weight_bounds=(0, 0.5, 1))

    def test_mvsk_tickers(self):
        mv = setup_efficient_mvsk()
        w = mv.min_mvsk()
        assert "GOOG" in w
        assert "AAPL" in w

    def test_mvsk_save_weights(self, tmp_path):
        mv = setup_efficient_mvsk()
        mv.min_mvsk()
        filepath = str(tmp_path / "weights.csv")
        mv.save_weights_to_file(filepath)
        import os

        assert os.path.exists(filepath)
