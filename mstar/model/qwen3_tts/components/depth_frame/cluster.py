"""Inline PTX for a CTA pair in a cluster of two: Gluon has no API for another CTA's shared memory.

The frame kernel is compiled for one CTA and launched in clusters of two. Each
CTA keeps a 4096-byte fp32 buffer and one mbarrier past Triton's shared memory,
at byte offset OFF of ``global_smem``. Rank 1 writes its partial tile into rank
0's buffer with asynchronous stores that count their bytes on rank 0's
mbarrier, so no release fence is needed; rank 0 expects the 4096 bytes and
waits on the barrier's phase.
"""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def smem_base(z):
    """The shared address of ``global_smem``, in ``z``'s layout."""
    return gl.inline_asm_elementwise("mov.u32 $0, global_smem;", "=r,r", [z], dtype=gl.int32, is_pure=True, pack=1)


@gluon.jit
def cluster_init(bar):
    """Thread 0 initializes the combine mbarrier at ``bar`` for one arrival; then the cluster syncs, so
    neither CTA's partner sends to it before it exists. ``bar``: one element per thread."""
    return gl.inline_asm_elementwise(
        "{ .reg .pred p; .reg .b32 t; mov.u32 t, %tid.x; setp.eq.u32 p, t, 0; "
        "@p mbarrier.init.shared::cta.b64 [$1], 1; fence.mbarrier_init.release.cluster; "
        "barrier.cluster.arrive.release; barrier.cluster.wait.acquire; mov.u32 $0, 0; }",
        "=r,r", [bar], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def cluster_sync(z):
    return gl.inline_asm_elementwise("barrier.cluster.arrive.release; barrier.cluster.wait.acquire; mov.u32 $0, 0;",
                                     "=r,r", [z], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def expect_tx(bar):
    """Thread 0 arrives on the local combine mbarrier, expecting the partner's 4096 bytes."""
    return gl.inline_asm_elementwise(
        "{ .reg .pred p; .reg .b32 t; mov.u32 t, %tid.x; setp.eq.u32 p, t, 0; "
        "@p mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], 4096; mov.u32 $0, 0; }",
        "=r,r", [bar], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def st_async4(addr, v, bar):
    """Rank 0's ``addr`` gets 4 consecutive fp32 elements of a thread in one 16-byte asynchronous store,
    counted on rank 0's mbarrier ``bar``. Addresses are this CTA's; mapa maps them to rank 0."""
    return gl.inline_asm_elementwise(
        "{ .reg .b32 ra, rb; mapa.shared::cluster.u32 ra, $4, 0; mapa.shared::cluster.u32 rb, $12, 0; "
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [ra], {$8, $9, $10, $11}, [rb]; "
        "mov.u32 $0, 0; mov.u32 $1, 0; mov.u32 $2, 0; mov.u32 $3, 0; }",
        "=r,=r,=r,=r,r,r,r,r,f,f,f,f,r,r,r,r", [addr, v, bar], dtype=gl.int32, is_pure=False, pack=4)


@gluon.jit
def wait_cluster(bar, phase):
    """Spin on the local mbarrier until ``phase`` completes, acquiring the partner's stores."""
    return gl.inline_asm_elementwise(
        "{ .reg .pred p; W${:uid}: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], $2; "
        "@!p bra W${:uid}; mov.u32 $0, 0; }",
        "=r,r,r", [bar, phase], dtype=gl.int32, is_pure=False, pack=1)


@gluon.jit
def ld_local4(addr):
    return gl.inline_asm_elementwise("ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,r,r,r,r", [addr],
                                     dtype=gl.float32, is_pure=False, pack=4)
