API Reference
=============

Use this generated reference to look up modules, classes, and functions under
``mstar/``. For requests sent to a running server, use :doc:`clients`.
The entries below come from module docstrings; expand a subpackage to find a symbol.

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
   mstar.observability
   mstar.streaming
   mstar.utils
   mstar.worker
