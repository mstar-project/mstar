# CUDA graphs after model porting

This reference belongs to a separate optimization task. Do not load or apply it
while completing the eager-first add-model MVP.

## Entry gate

Begin capture work only after the native eager path passes component, node,
graph/resource, and live-serving correctness gates, and after a benchmark has
recorded an observational baseline. Keep that eager path reachable as the
correctness control.

## Optimization scope

Use the checked-out `CudaGraphConfig` and `PiecewiseCudaGraphConfig` contracts.
Choose whole-forward capture when the complete node has stable shapes and
addresses. Choose piecewise capture only when a stable inner region provides a
clear benefit while the surrounding control flow must remain eager.

Resource-backed state must remain engine-owned. Captured forwards still declare
resource work, use engine-leased slots, and read stable buffers owned by the
resource. Capture is not permission to add a model-owned pool or change generic
engine lifecycle semantics.

## Required evidence

Compare captured output with the eager control, test every replay bucket and
fallback path, and verify that capture failure returns to eager execution. Then
measure the same benchmark schema used by the porting task so before/after
results remain comparable.
