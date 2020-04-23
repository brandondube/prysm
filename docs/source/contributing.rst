************
Contributing
************

This document summarizes how to contribute to prysm.  The basic steps are:

* Fork the project on GitHub

* Clone your fork

* Add the main repository as a remote :code:`git remote add upstream https://github.com/brandondube/prysm/prysm.git`

* Most prysm development is done on the :code:`dev` branch, so work from there:

.. code-block :: bash

    git checkout dev
    git pull upstream dev

* Track your changes, ideally with "atomic" commits -- one commit per logical change (multiple files are OK, but if your commit crosses +/- 1000 lines, it probably should have been several commits).

* When ready, push your changes to your fork on GitHub and open a Pull Request (PR).  Reference any relevant issues in the body of the PR.

* Open an issue on the main prysm repository with any problems or inquiries.


Guidelines
==========

* prysm uses `numpy style docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.  All changes should include updated docstrings, or fresh docstrings for new features.

* Do not include shebangs or other meta-comments at the top of files.

* prysm uses unix-style line endings.  Git can be configured to convert CRLF to LF for you.

* PRs should update tests or introduce new tests as needed to maintain coverage and correctness.

* Use :code:`from prysm.mathops import e` instead of :code:`import numpy as np`, using the :code:`e` object as you would the :code:`np` one.  You cannot import from :code:`e`, so for example you may have to write :code:`e.fft.fft2`.

* If your code creates new arrays, please maintain conformance with prysm's precision options:

.. code-block :: python

    from .conf import config

    ary = e.arange(lower, upper, spacing, dtype=config.precision)


For a not-all-inclusive list of welcome contributions, please see the `open issues <https://github.com/brandondube/prysm/issues>`_.
