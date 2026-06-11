API Reference
=============

This reference is generated automatically from the docstrings in each module. It mirrors
the package layout under ``mstar/``; expand a subpackage to drill down to individual
modules, classes, and functions.

.. note::

   Because the reference is built by importing each module to read its docstrings, the
   build must run in the project's runtime environment (``pip install -e ".[dev]"`` plus
   ``docs/requirements.txt``) so that ``torch`` and the other core dependencies are
   importable. See ``docs/conf.py`` for details.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   mstar.api_server
   mstar.communication
   mstar.conductor
   mstar.engine
   mstar.graph
   mstar.model
   mstar.streaming
   mstar.utils
   mstar.worker
