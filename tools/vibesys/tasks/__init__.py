"""Task-type registry.

Importing this package registers every built-in task type in ``REGISTRY``.
Add a task type by creating ``tasks/<name>/`` with a ``TaskType`` subclass and
appending one ``register(...)`` line here.
"""

from __future__ import annotations

from tools.vibesys.tasks.add_model.task import AddModelTask
from tools.vibesys.tasks.base import REGISTRY, register

register(AddModelTask())

__all__ = ["REGISTRY", "register"]
