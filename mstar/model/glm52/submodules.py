"""NodeSubmodule for the GLM-5.2 LLM node — interface skeleton.

Deliberately unimplemented: the compute lands with components/ (decoder
stack, MoE, weight loading) and the MLA attention path depends on the
paged-MLA engine sequencing decision (users/garv/kimik27-integration).
The scaffold's CPU tests exercise the Model contract in dummy mode and
never instantiate this class.

Implementation map (per docs/adding_models.rst step 4/5):
    prepare_inputs   - token ids / seq-len bookkeeping into ARNodeInputs
    preprocess       - collate a continuous batch; FlashInfer plan
    forward          - embed -> 78 decoder layers -> lm_head -> sample
    forward_batched  - the batched twin (can_batch=True for decode)
    check_stop       - stop "decode_loop" on any of config.eos_token_ids
    get_cuda_graph_configs - decode: BasicBatched; prefill: FlashInferPacked
"""

from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.submodule_base import ARNodeSubmodule


class Glm52LLMSubmodule(ARNodeSubmodule):
    """Embed + decoder stack + lm_head for GLM-5.2 (not yet implemented)."""

    def __init__(self, language_model, config: Glm52ModelConfig):
        raise NotImplementedError(
            "Glm52LLMSubmodule lands with mstar/model/glm52/components/ — "
            "see the implementation map in this module's docstring.",
        )
