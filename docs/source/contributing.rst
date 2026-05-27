************
Contributing
************

This document summarizes how to contribute to prysm. The basic steps are:

* Fork the project on GitHub

* Clone your fork

* Add the main repository as a remote:

.. code-block:: bash

    git remote add upstream https://github.com/brandondube/prysm.git


* Track your changes, ideally with "atomic" commits -- one commit per logical
change (multiple files are OK, but if your commit crosses +/- 1000 lines, it
probably should have been several commits).

* When ready, push your changes to your fork on GitHub and open a Pull Request
(PR). Reference any relevant issues in the body of the PR.

* Open an issue on the main prysm repository with any problems or inquiries.


Guidelines
==========

* prysm uses `numpy style docstrings
<https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.
All changes should include updated docstrings, or fresh docstrings for new
features.

* Do not include shebangs or other meta-comments at the top of files.

* prysm uses unix-style line endings. Git can be configured to convert CRLF to
LF for you.

* PRs should update tests or introduce new tests as needed to maintain coverage
and correctness.

For mathematical libraries, import them from :code:`prysm.mathops`. These
include:

.. code-block:: python

    from prysm.mathops import np, fft, interpolate, ndimage

prysm's backend can be changed at will by the user. Importing this way avoids
locking the user into numpy or scipy.

* If your code creates new arrays, please maintain conformance with prysm's
precision options:

.. code-block :: python

    from prysm.conf import config

    ary = np.arange(lower, upper, spacing, dtype=config.precision)


Building the Documentation
==========================

The documentation is built with Sphinx and nbsphinx. Use the project conda
environment when checking docs locally:

.. code-block:: bash

    python -m sphinx -b html -n -W --keep-going docs/source docs/build/html

The docs Makefile provides the same strict build check:

.. code-block:: bash

    cd docs
    make strict-html

Warnings should be treated as build failures. Notebook execution errors should
also fail the build; fix the notebook or the underlying library behavior instead
of hiding tracebacks in rendered docs.


Notebook Documentation
======================

Tutorials, how-tos, and explanations can be authored as notebooks when the
reader benefits from executable examples and rendered plots. Keep notebooks
small enough to run during a docs build, give each notebook exactly one clear
top-level heading, and prefer library helpers over one-off documentation-only
helpers.

When linking between documentation pages, prefer Sphinx references such as
:code:`:doc:\`../tutorials/First Diffraction Model\`` from RST files. Avoid
hard-coded Markdown links to notebook filenames when a Sphinx cross-reference
can express the relationship.


Adding API Pages
================

Add new API pages under :code:`docs/source/api` and include them in
:code:`docs/source/api/index.rst`. Start each page with a short summary of what
the module is for, followed by the appropriate :code:`automodule` directive.
