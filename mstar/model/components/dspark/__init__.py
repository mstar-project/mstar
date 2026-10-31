"""DSpark: a drafter that reads the target's layers and drafts a block in one pass (see
``drafter.py``), configured per checkpoint by ``DSparkConfig``."""

from mstar.model.components.dspark.config import DSparkConfig
from mstar.model.components.dspark.drafter import DSpark, DSparkDrafter, DSparkKeys

__all__ = ["DSpark", "DSparkConfig", "DSparkDrafter", "DSparkKeys"]
