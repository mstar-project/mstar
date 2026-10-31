"""One classifier-free guidance branch per rank of a lockstep instance.

A guided denoise step is two transformer rows that never read each other: the
conditional and the unconditional forward share only their inputs. With
``cfg_size: 2`` on a node group, rank 0 of the group runs every request's
conditional row and rank 1 its unconditional row, each at the shape one branch
has on one GPU, and one all-gather hands both ranks both velocities. Both then
run the identical solver step, so the latents stay replicated and the exchange
is the only traffic between the ranks.

The placement is opt-in per node (``ShardingConfig.cfg_enabled_nodes``) and the
engine hands it to ``Model.get_submodule`` as ``guidance_parallel`` only when
the node's group asks for it. A step with no second branch to exchange (guidance
off) stays symmetric without it: both ranks run the same row and nothing is sent.
"""

import torch

from mstar.distributed.communication import CommGroup


class GuidanceParallel:
    """The rank's share of a guided step and the exchange that completes it."""

    # The two branches, in the order an all-gather returns them.
    NUM_BRANCHES = 2

    def __init__(self, group: CommGroup):
        if group.world_size != self.NUM_BRANCHES:
            raise ValueError(
                f"Guidance parallelism places {self.NUM_BRANCHES} branches, one per rank; "
                f"the group over ranks {group.group_members} has {group.world_size}"
            )
        self.group = group

    @property
    def runs_cond(self) -> bool:
        """This rank runs the conditional rows; the other rank the unconditional ones."""
        return self.group.rank == 0

    def exchange(self, velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(cond, uncond)``, each shaped like ``velocity``, this rank's rows.

        Collective, and it replays from a captured graph, so the caller must run
        it for every guided step on both ranks."""
        cond, uncond = self.group.all_gather(velocity.contiguous(), dim=0).chunk(self.NUM_BRANCHES)
        return cond, uncond
