"""
The ``exceptions`` module houses custom exceptions. Currently implemented:

- OptimizationError
"""


class OptimizationError(Exception):
    """
    When an optimization routine fails – usually, this means
    that cvxpy has not returned the "optimal" flag.
    """

    def __init__(self, *args, **kwargs):
        default_message = (
            "Please check your objectives/constraints or use a different solver."
        )
        super().__init__(default_message, *args, **kwargs)


class InstantiationError(Exception):
    """
    Errors related to the instantiation of pypfopt objects, e.g adding constraints to an
    already-solved problem
    """

    pass
