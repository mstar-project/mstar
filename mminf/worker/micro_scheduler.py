from dataclasses import dataclass

from mminf.engine.base import EngineType
from mminf.graph.base import GraphStage
from mminf.worker.engine_manager import EngineManager
from mminf.worker.stage_manager_utils import SubgraphsManager


@dataclass
class ReadyStageEntry:
    """A ready stage entry for a single request."""
    request_id: str
    subgraph_id: str
    phase: str


@dataclass
class ScheduledBatch:
    """A batch of stages ready to be executed."""
    stage_name: str
    phase: str
    request_ids: list[str]
    stages: list[GraphStage]  # the popped GraphStage objects


# Priority: lower value = higher priority
# AR decode is most latency-sensitive
PRIORITY = {
    EngineType.AR: 0,
    EngineType.FLOW: 1,
    EngineType.ENC_DEC: 2,
}


class MicroScheduler:
    """
    Simple MVP scheduler: scans all subgraph queues for ready stages,
    groups by stage name, returns the highest-priority group.
    """

    def __init__(self, engine_manager: EngineManager):
        self.engine_manager = engine_manager

    def get_next_batch(
        self, subgraphs_manager: SubgraphsManager
    ) -> ScheduledBatch | None:
        """
        Scans all subgraph queues for ready stages.
        Groups by stage name. Returns highest-priority group.
        """
        # Collect all ready (stage_name, request_id, phase) tuples
        # grouped by stage name
        stage_name_to_requests: dict[str, list[ReadyStageEntry]] = {}

        for subgraph_id, queue in subgraphs_manager.queues.items():
            ready_map = queue.get_ready_stage_names()
            for request_id, stage_names in ready_map.items():
                phase = subgraphs_manager.get_phase(request_id)
                for sname in stage_names:
                    stage_name_to_requests.setdefault(sname, []).append(
                        ReadyStageEntry(request_id, subgraph_id, phase)
                    )

        if not stage_name_to_requests:
            return None

        # Pick the stage name with highest priority (lowest PRIORITY value)
        best_stage_name = None
        best_priority = float("inf")

        for sname in stage_name_to_requests:
            if sname not in self.engine_manager.stage_to_engine:
                continue
            engine = self.engine_manager.get_engine(sname)
            prio = PRIORITY.get(engine.engine_type(), 99)
            if prio < best_priority:
                best_priority = prio
                best_stage_name = sname

        if best_stage_name is None:
            return None

        # Pop ready stages for all requests of this stage name
        entries = stage_name_to_requests[best_stage_name]
        request_ids = []
        all_stages = []
        phase = entries[0].phase

        for entry in entries:
            queue = subgraphs_manager.queues[entry.subgraph_id]
            popped = queue.pop_ready_stages(entry.request_id, [best_stage_name])
            if popped:
                request_ids.append(entry.request_id)
                all_stages.extend(popped)
                phase = entry.phase

        if not request_ids:
            return None

        return ScheduledBatch(
            stage_name=best_stage_name,
            phase=phase,
            request_ids=request_ids,
            stages=all_stages,
        )
