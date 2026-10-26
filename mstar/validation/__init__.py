"""One pass over a deployment, which the conductor runs before it spawns a worker.

``validate_deployment`` runs every startup check that needs only the model,
the serving YAML and the environment, and reports every problem it finds at
once. The checks left on the worker need what only the worker builds: its
communicator, its built resources and capture results, and live process
groups across ranks.
"""

from collections.abc import Mapping
from typing import Any, NamedTuple

from mstar.distributed.base import ShardingConfig
from mstar.engine.resources.lora.serving import refuse_unresolved_lora_adapters
from mstar.model.base import Model, WorkerGraph, built_resources, enabled_resources, parse_autocast_dtype
from mstar.validation.knobs import check_knobs
from mstar.validation.placement import check_worker_graphs, node_groups_problems
from mstar.validation.resources import check_resources, resources_shape_problem


class DeploymentError(ValueError):
    """Every problem ``validate_deployment`` found, one per line."""

    def __init__(self, config_path: str, problems: list[str]):
        self.config_path = config_path
        self.problems = problems
        count = f"{len(problems)} problem{'' if len(problems) == 1 else 's'}"
        listed = "\n".join(f"  - {problem}" for problem in problems)
        super().__init__(f"{config_path}: {count}; no worker was started:\n{listed}")


# What a model's YAML reader raises on a value of the wrong shape (a list
# where it reads a mapping, say): reported with the rest, not raised past them.
_UNREADABLE = (TypeError, AttributeError, KeyError, IndexError)


def _unreadable(step: str, e: Exception) -> str:
    return f"{step} could not read the config: {type(e).__name__}: {e}"


def _checked(step: str, check, *args) -> list[str]:
    """``check(*args)``'s problems; a model hook it calls may read checkpoint
    files (FLUX.2 klein's walk graphs), and that failure is one more problem."""
    try:
        return check(*args)
    except (ValueError, OSError) as e:
        return [f"{step} failed: {type(e).__name__}: {e}"]
    except _UNREADABLE as e:
        return [_unreadable(step, e)]


class Deployment(NamedTuple):
    worker_graphs: list[WorkerGraph]
    sharding_config: ShardingConfig


def validate_deployment(
    model: Model, config: Mapping[str, Any], config_path: str,
) -> Deployment:
    """Check the parsed YAML at ``config_path`` against ``model`` and return
    the worker graphs and sharding config to deploy.

    Raises ``DeploymentError`` listing every problem. The checks on worker
    graphs run only once ``node_groups`` places every walk, since the graphs
    cannot be built before that.
    """
    if not isinstance(config, Mapping):
        raise DeploymentError(config_path, [f"the config must be a YAML mapping; got {config!r}"])
    problems = list(check_knobs())
    if "max_seq_len" not in config:
        problems.append("the config must define `max_seq_len`")
    try:
        parse_autocast_dtype(config.get("autocast_dtype"))
    except ValueError as e:
        problems.append(str(e))
    # model hooks read `resources:` as a mapping; check_resources reports a bad one
    resources_readable = resources_shape_problem(config) is None
    if resources_readable:
        try:
            model.validate_config_yaml(config, config_path)
        except ValueError as e:
            problems.append(str(e))
        except OSError as e:
            problems.append(f"validate_config_yaml failed: {type(e).__name__}: {e}")
        except _UNREADABLE as e:
            problems.append(_unreadable("validate_config_yaml", e))
        try:
            refuse_unresolved_lora_adapters(config)
        except ValueError as e:
            problems.append(str(e))
        except _UNREADABLE as e:
            problems.append(_unreadable("refuse_unresolved_lora_adapters", e))

    malformed = node_groups_problems(config)
    problems += malformed
    deployment = None
    if not malformed:
        # refuses the placement (check_placement) once the model has read it
        try:
            worker_graphs = model.get_worker_graphs(config_path)
            sharding_config = model.get_sharding_config(config_path)
        except DeploymentError as e:
            problems += e.problems
        except ValueError as e:
            problems.append(str(e))
        except OSError as e:
            problems.append(f"reading node_groups and sharding_config failed: {type(e).__name__}: {e}")
        except _UNREADABLE as e:
            problems.append(_unreadable("reading node_groups and sharding_config", e))
        else:
            deployment = Deployment(worker_graphs, sharding_config)
    # After get_worker_graphs: a model may declare resources by placement
    # (BAGEL's disaggregated prefill), and workers get the model as it is now.
    # a model may read checkpoint files here (LTX-2.5): its failure joins the
    # problems found so far, and only the checks that need the specs are skipped
    specs = None
    try:
        # a `resources:` block check_resources refuses turns no resource on
        specs = enabled_resources(model, config if resources_readable else {**config, "resources": None})
    except (ValueError, OSError) as e:
        problems.append(
            f"declaring the node resources failed: {type(e).__name__}: {e}; "
            "the `resources:` checks, which need the declarations, were skipped"
        )
    except _UNREADABLE as e:
        problems.append(_unreadable("get_node_resources", e))
    if specs is not None:
        problems += _checked("checking the node resources", check_resources, model, specs, config)
        # the placement checks see what the workers build of them
        try:
            specs = built_resources(specs)
        except (ValueError, OSError) as e:
            problems.append(f"building the node resources failed: {type(e).__name__}: {e}")
            specs = None
    if deployment is not None:
        problems += _checked(
            "checking the worker graphs", check_worker_graphs,
            model, deployment.worker_graphs, deployment.sharding_config, specs,
        )

    if problems:
        raise DeploymentError(config_path, problems)
    return deployment
