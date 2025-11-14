def test_import_modules():
    from pypfopt import (
        base_optimizer,
        black_litterman,
        cla,
        discrete_allocation,
        exceptions,
        expected_returns,
        hierarchical_portfolio,
        objective_functions,
        plotting,
        risk_models,
    )


def test_explicit_import():
    from pypfopt.black_litterman import (
        BlackLittermanModel,
        market_implied_prior_returns,
        market_implied_risk_aversion,
    )
    from pypfopt.cla import CLA
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    from pypfopt.efficient_frontier import (
        EfficientCVaR,
        EfficientFrontier,
        EfficientSemivariance,
    )
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.risk_models import CovarianceShrinkage


def test_import_toplevel():
    from pypfopt import (
        CLA,
        BlackLittermanModel,
        CovarianceShrinkage,
        DiscreteAllocation,
        EfficientCVaR,
        EfficientFrontier,
        EfficientSemivariance,
        HRPOpt,
        get_latest_prices,
        market_implied_prior_returns,
        market_implied_risk_aversion,
    )
