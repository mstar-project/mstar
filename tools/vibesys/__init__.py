"""mstar-vibesys: task-scoped utilities that drive VibeSys against the mstar repo.

VibeSys (``pip install vibesys``) is a generic agentic build/optimize loop. This
package is the mstar-side glue: for each *task type* (adding a model, optimizing a
walk, ...) it renders a VibeSys input bundle, prepares an isolated git-worktree
seed of mstar plus a reproducible Docker eval image, and hands off to the
installed ``vibesys`` CLI.

The framework is a thin registry of task types over shared machinery in
``core/``. Adding a new task type = dropping a module under ``tasks/`` and
registering it; ``core/`` does not change. ``add-model`` is the first entry.
"""

from __future__ import annotations

__version__ = "0.0.1"
