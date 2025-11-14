# Contributing to PyPortfolioOpt

Please refer to the roadmap for a list of areas that I think PyPortfolioOpt could really benefit
from. In addition, the following is always welcome::

- Improve performance of existing code (but not at the cost of readability) – are there any nice numpy tricks I've missed?
- Add new optimization objective functions. For example, if you think that the best performance metric has not been included, write it into a function (or suggest it in [Issues](https://github.com/pyportfolio/pyportfolioopt/issues) and I will have a go).
- Help me write more tests! If you are someone learning about quant finance and/or unit testing in python, what better way to practice than to write some tests on an open-source project! Feel free to check for edge cases, or test performance on a dataset with more stocks.

## Guidelines

### Seek early feedback

Before you start coding your contribution, it may be wise to raise an issue on GitHub to discuss whether the contribution is appropriate for the project.

I care a lot about having a clean API, so I am unlikely to accept a random PR that significantly complicates the API (or the dependencies).

### Code style

For this project I have used [Black](https://github.com/ambv/black) as the formatting standard, with all of the default arguments. I recommend that PRs use black.

### Testing

Any contributions **must** be accompanied by unit tests (written with `pytest`). These are incredibly simple to write, just find the relevant test file (or create a new one), and write a bunch of `assert` statements. The test should be applied to the dummy dataset I have provided in `tests/stock_prices.csv`, and should cover core functionality, warnings/errors (check that they are raised as expected), and limiting behaviour or edge cases.

### Documentation

Inline comments (and docstrings!) are great when needed, but don't go overboard. A lot of the explanation can and should be offloaded to ReadTheDocs. Docstrings should follow [PEP257](https://stackoverflow.com/questions/2557110/what-to-put-in-a-python-module-docstring) semantically and sphinx syntactically.

I would appreciate if changes are accompanied by relevant documentation – it doesn't have to be pretty, because I will probably try to tidy it up before it goes onto ReadTheDocs, but it'd make things a lot simpler to have the person who wrote the code explain it in their own words.

## Questions

If you have any questions related to the project, it is probably easiest to [raise an issue](https://github.com/pyportfolio/pyportfolioopt/issues), and I will tag it as a question.

If you have questions unrelated to the project, drop me an email – contact details can be found on my [website](https://reasonabledeviations.com/about/)

## Bugs/issues

If you find any bugs or the portfolio optimization is not working as expected, feel free to [raise an issue](https://github.com/pyportfolio/pyportfolioopt/issues). I would ask that you provide the following information in the issue:

- Descriptive title so that other users can see the existing issues
- Operating system, python version, and python distribution (optional).
- Minimal example for reproducing the issue.
- What you expected to happen
- What actually happened
- A full traceback of the error message (omit personal details as you see fit).
