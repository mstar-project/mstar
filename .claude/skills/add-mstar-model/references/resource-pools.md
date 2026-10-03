# Engine-owned resource pools

Read this when a port has state that is pooled, allocated, capacity-limited,
stable across calls, dependency-ordered, or explicitly cleaned up.

## Ownership decision

| state behavior | representation |
|---|---|
| Produced by one graph node and consumed by another or a later Walk | A named `GraphEdge`, using `persist=True` only when graph routing must retain it. |
| Request-local metadata or tensors with no allocator, pool, resident limit, or shared stable-address contract | Engine-provided per-request state. |
| Shared storage with admission, slots/pages, resident capacity, stable addresses, build dependencies, publish/retrieve, eviction, or cleanup rules | An engine-owned `Resource`. |

Do not keep the third category in a submodule dictionary, module buffer pool, or
custom cleanup loop. That creates a lifecycle the scheduler cannot admit,
backpressure, reclaim, or reliably release.

## Reuse before adding

Map required behavior to the existing resources first:

- paged autoregressive state: KV, attention, and position resources;
- fixed encoder context: cross-attention and its KV resource;
- packed cacheless encoder attention: ragged attention;
- token selection and request sampling parameters: sampler; and
- ordinary tensors between stages: graph edges rather than a resource.

Use the resource keys already established by the closest reference model when
their meanings match. A resource key must agree across the model spec,
deployment overrides, node step declaration, and forward lookup.

## Add a resource kind only when required

A new kind stays inside `mstar/engine/resources/<kind>/` and uses the existing
interfaces:

1. Add a `NodeResourceSpec` subclass describing process-lifetime shape and
   capacity. Its `resource_class` lazily selects the concrete builder, and its
   `depends_on()` names prerequisite resource specs supplied to `build()`.
2. Add a `ResourceReqConfig` subclass only for per-request knobs needed during
   `ingest_request()`.
3. Add a `ResourceStep` subclass carrying the host-side facts needed by one
   execution step.
4. Implement a `Resource` manager with `build()` and only the lifecycle hooks it
   needs: request ingest/removal, step admit/plan/commit, optional
   publish/retrieve or eviction, graph buffers, validation, and cleanup. Its
   `depends_on()` names the same prerequisite keys when step planning must run
   after them.
5. Export declaration types through `mstar/engine/resources/__init__.py` and add
   focused CPU tests for dependency order, capacity, cleanup, and invalid use.

The model then declares the spec in `get_node_resources()`, returns any request
config from `get_request_resource_configs()`, declares each step from the node,
and reads the built manager through `engine_inputs.resources` or a bound layer.

Do not change the generic engine, base resource, runner, or step-envelope
lifecycle to special-case the model. The absence of a ready-made resource kind
is not a blocker: add one through the interfaces above and preserve every
engine lifecycle invariant. Report an evidence-backed blocker only when a
concrete attempted mapping exposes a required capability missing from those
generic extension interfaces. Name the exact interface and operation that are
missing. A blocker stops that direction; it never authorizes a model-owned
pool, allocator, capacity limit, stable buffer, dependency scheduler, or
cleanup lifecycle as a workaround or prototype.

## Lifecycle invariants

The generic runner owns:

```text
ingest request
  -> declare step
  -> admit
  -> plan in dependency order
  -> preprocess and forward
  -> commit
  -> remove request
  -> process cleanup
```

`admit()` must fail without partially claiming capacity. `remove_request()` must
return every claim, including after failure or cancellation. `plan()` should
stage fixed-address data and return immutable information for downstream
resources through `StepContext.plan_results`. `commit()` records only work that
successfully ran.

## Resident capacity is not step batching

Document both quantities independently:

- resident capacity: how many request sessions or allocations remain live; and
- step batch size: how many requests one forward processes together.

A fixed pool can retain N sessions while a node executes one session per step.
Do not derive one value from the other, and make deployment overrides describe
the resident quantity explicitly.

## Waypoint ring-resource case study

[Waypoint draft PR #251](https://github.com/mstar-project/mstar/pull/251) is the
custom fixed-ring example. Use its current design ideas, not historical commits
where model code owned persistent pool behavior:

- the DiT node declares a ring KV spec and attention dependency;
- `num_worlds` sizes resident session slots, while `max_batch_size` remains a
  separate one-step limit;
- request admission claims a world, planning stages its stable world index, and
  request removal resets and releases the world;
- a resource-specific step carries the ring clock and rejects discontinuity;
  and
- fixed layer geometry comes from the checkpoint/model declaration, while the
  deployment may tune only resident world count.

The draft may contain broader engine edits or capture work that are outside an
add-model MVP. Copy the resource boundary and testable invariants only. Implement
the port eager-first and report any missing core hook rather than importing the
draft's historical lifecycle changes.
