############
Contributing
############

Some of the things that I'd love for people to help with:

- Improve performance of existing code (but not at the cost of readability)
- Add new optimization objectives. For example, if you would like to use something other
  than the Sharpe ratio, write an optimizer! (or suggest it in
  `Issues <https://github.com/pyportfolio/pyportfolioopt/issues>`_ and I will have a go).
- Help me write more tests! If you are someone learning about quant finance and/or unit
  testing in python, what better way to practice than to write some tests on an
  open-source project! Feel free to check for edge cases, or for uncommon parameter
  combinations which may cause silent errors.


Guidelines
==========

Seek early feedback
-------------------

Before you start coding your contribution, it may be wise to
`raise an issue <https://github.com/pyportfolio/pyportfolioopt/issues>`_ on
GitHub to discuss  whether the contribution is appropriate for the project.

Code style
----------

For this project I have used `Black <https://github.com/ambv/black>`_ as the
formatting standard, with all of the default settings. It would be much
appreciated if any PRs follow this standard because if not I will have to
format before merging.

Testing
-------

Any contributions **must** be accompanied by unit tests (written with ``pytest``).
These are incredibly simple to write, just find the relevant test file (or create
a new one), and write a bunch of ``assert`` statements. The test should be applied
to the dummy dataset I have provided in ``tests/stock_prices.csv``, and should
cover core functionality, warnings/errors (check that they are raised as expected),
and limiting behaviour or edge cases.

Documentation
-------------

Inline comments are great when needed, but don't go overboard. Docstring content
should follow `PEP257 <https://stackoverflow.com/questions/2557110/what-to-put-in-a-python-module-docstring>`_
semantically and sphinx syntactically, such that sphinx can automatically document
the methods and their arguments. I am personally not a fan of writing long
paragraphs in the docstrings: in my view, docstrings should state briefly how an
object can be used, while the rest of the explanation and theoretical background
should be offloaded to ReadTheDocs.

I would appreciate if changes are accompanied by relevant documentation - it
doesn't have to be pretty, because I will probably try to tidy it up before it
goes onto ReadTheDocs, but it'd make things a lot simpler to have the person who
wrote the code explain it in their own words.

Questions
=========

If you have any questions related to the project, it is probably best to
`raise an issue <https://github.com/pyportfolio/pyportfolioopt/issues>`_ and
I will tag it as a question.

If you have questions *unrelated* to the project, drop me an email - contact
details can be found on my `website <https://reasonabledeviations.com/about/>`_.

Bugs/issues
===========

If you find any bugs or the portfolio optimization is not working as expected,
feel free to `raise an issue <https://github.com/pyportfolio/pyportfolioopt/issues>`_.
I would ask that you provide the following information in the issue:

- Descriptive title so that other users can see the existing issues
- Operating system, python version, and python distribution (optional).
- Minimal example for reproducing the issue.
- What you expected to happen
- What actually happened
- A full traceback of the error message (omit personal details as you see fit).
