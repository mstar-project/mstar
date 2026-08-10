"""Parse a TorchInductor debug dump into per-kernel work estimates.

Input: the directory ``torch._inductor.config.trace.debug_dir`` points at (one
``output_code.py`` + ``fx_graph_readable.py`` per compiled subgraph). Model
agnostic — nothing here knows what model produced the dump.

What we extract per subgraph:
  - every fused Triton kernel: the exact input/output tensors (shape + dtype)
    from the "Graph fragment" comment Inductor emits above each kernel def,
    its fused ATen ops, load/store/reduction counts, and the source lines it
    fuses (fx node -> ``# File: path:line`` comments in fx_graph_readable.py);
  - every ``extern_kernels.*`` call (cuBLAS/library GEMMs) with resolved
    argument shapes/dtypes, so callers can compute exact FLOPs;
  - the subgraph boundary: bytes of external inputs and of newly-materialized
    output buffers (graph-break traffic).
"""

from __future__ import annotations

import glob
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from math import prod

_DTYPE_BYTES = {
    "bf16": 2, "f16": 2, "f32": 4, "f64": 8,
    "i8": 1, "u8": 1, "b8": 1, "i16": 2, "i32": 4, "i64": 8,
    "f8e4m3fn": 1, "f8e5m2": 1, "f8e4m3fnuz": 1, "f8e5m2fnuz": 1,
    "torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4, "torch.float64": 8,
    "torch.int8": 1, "torch.uint8": 1, "torch.bool": 1, "torch.int16": 2,
    "torch.int32": 4, "torch.int64": 8,
    "torch.float8_e4m3fn": 1, "torch.float8_e5m2": 1,
}


def dtype_nbytes(dtype: str) -> int:
    return _DTYPE_BYTES.get(dtype, 2)


@dataclass(frozen=True)
class TensorInfo:
    shape: tuple[int, ...]
    dtype: str

    @property
    def numel(self) -> int:
        return prod(self.shape) if self.shape else 1

    @property
    def nbytes(self) -> int:
        return self.numel * dtype_nbytes(self.dtype)


@dataclass
class TritonKernel:
    name: str
    aten_ops: set[str] = field(default_factory=set)
    src_nodes: set[str] = field(default_factory=set)
    inputs: dict[str, TensorInfo] = field(default_factory=dict)     # placeholder -> tensor
    outputs: dict[str, TensorInfo] = field(default_factory=dict)    # returned node -> tensor
    op_of_node: dict[str, str] = field(default_factory=dict)        # fragment node -> aten op
    node_tensors: dict[str, TensorInfo] = field(default_factory=dict)
    num_load: int | None = None
    num_store: int | None = None
    num_reduction: int | None = None
    source_lines: "OrderedDict[str, str]" = field(default_factory=OrderedDict)

    @property
    def traffic_bytes(self) -> int:
        """Unique input bytes read + output bytes written (one pass each)."""
        return sum(t.nbytes for t in self.inputs.values()) + sum(t.nbytes for t in self.outputs.values())


@dataclass
class ExternCall:
    op: str                                   # mm / addmm / bmm / ...
    aten_ops: set[str] = field(default_factory=set)
    src_nodes: set[str] = field(default_factory=set)
    args: list[TensorInfo | None] = field(default_factory=list)
    out: TensorInfo | None = None
    source_lines: "OrderedDict[str, str]" = field(default_factory=OrderedDict)

    def gemm_shape(self) -> tuple[int, int, int, int] | None:
        """(batch, m, n, k), or None if this call isn't a recognized GEMM."""
        mats = [a for a in self.args if a is not None and len(a.shape) >= 2]
        if self.op == "addmm" and len(mats) >= 2:
            mats = mats[-2:]                  # drop the bias operand
        if len(mats) < 2:
            return None
        a, b = mats[0], mats[1]
        if self.op in ("mm", "addmm") and len(a.shape) == 2 and len(b.shape) == 2:
            (m, k), (_, n) = a.shape, b.shape
            return (1, m, n, k)
        if self.op in ("bmm", "baddbmm") and len(a.shape) == 3 and len(b.shape) == 3:
            (bs, m, k), (_, _, n) = a.shape, b.shape
            return (bs, m, n, k)
        return None

    @property
    def flops(self) -> int | None:
        s = self.gemm_shape()
        if s is None:
            return None
        bs, m, n, k = s
        return 2 * bs * m * n * k

    @property
    def traffic_bytes(self) -> int | None:
        s = self.gemm_shape()
        if s is None:
            return None
        real = [a for a in self.args if a is not None]
        total = sum(a.nbytes for a in real)
        if self.out is not None:
            total += self.out.nbytes
        return total

    @property
    def dtype(self) -> str:
        for a in self.args:
            if a is not None:
                return a.dtype
        return "bf16"


@dataclass
class Subgraph:
    name: str
    path: str
    triton: dict[str, TritonKernel] = field(default_factory=dict)
    externs: list[ExternCall] = field(default_factory=list)
    input_tensors: dict[str, TensorInfo] = field(default_factory=dict)
    output_tensors: list[TensorInfo] = field(default_factory=list)  # newly-written bufs returned
    node_source: dict[str, tuple[str, str, str]] = field(default_factory=dict)  # node -> (file, line, code)

    @property
    def input_bytes(self) -> int:
        return sum(t.nbytes for t in self.input_tensors.values())

    @property
    def output_bytes(self) -> int:
        return sum(t.nbytes for t in self.output_tensors)


# ---- fx_graph_readable.py ----
_FX_FILE_RE = re.compile(r"#\s*File:\s*(.+?):(\d+)\s+in\s+\S+,\s*code:\s*(.*)")
_FX_NODE_RE = re.compile(r'^\s*(\w+)\s*:\s*"(\w+)\[([\d, ]*)\]"(.*)')
_FX_SIG_RE = re.compile(r'(\w+)\s*:\s*"(\w+)\[([\d, ]*)\]"')
_GEMM_TARGETS = ("mm", "addmm", "bmm", "baddbmm", "convolution")


def _parse_shape(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.replace(" ", "").split(",") if x)


def parse_fx_readable(
    path: str,
) -> tuple[dict[str, tuple[str, str, str]], dict[str, TensorInfo], list[tuple[str, str]]]:
    """Returns (node -> (basename, line, code), name -> TensorInfo,
    ordered [(gemm_node_name, base_op)] for pairing with extern_kernels calls)."""
    node_src: dict[str, tuple[str, str, str]] = {}
    tensors: dict[str, TensorInfo] = {}
    gemm_nodes: list[tuple[str, str]] = []
    if not os.path.exists(path):
        return node_src, tensors, gemm_nodes
    cur: tuple[str, str, str] | None = None
    for line in open(path):
        f = _FX_FILE_RE.search(line)
        if f:
            cur = (os.path.basename(f.group(1)), f.group(2), f.group(3).strip())
            continue
        if "def forward(" in line:
            for name, dt, shape in _FX_SIG_RE.findall(line):
                tensors[name] = TensorInfo(_parse_shape(shape), dt)
            continue
        n = _FX_NODE_RE.match(line)
        if n:
            tensors[n.group(1)] = TensorInfo(_parse_shape(n.group(3)), n.group(2))
            if cur:
                node_src[n.group(1)] = cur
            for op in _GEMM_TARGETS:
                if f"torch.ops.aten.{op}." in n.group(4):
                    gemm_nodes.append((n.group(1), op))
                    break
    return node_src, tensors, gemm_nodes


# ---- output_code.py ----
_FRAG_NODE_RE = re.compile(
    r'^#\s+%(\w+)\s*:\s*Tensor\s*"(\w+)\[([\d, ]*)\]\[[\d, ]*\]\S*"(?:\[num_users=\d+\])?\s*=\s*'
    r"(PlaceHolder\[target=(\w+)\]|call_function\[target=([\w.]+)\])"
)
_FRAG_RET_RE = re.compile(r"^#\s+return\s+(.+)$")
_SRC_RE = re.compile(r"Source Nodes:\s*\[(.*?)\],\s*Original ATen:\s*\[(.*?)\]")
_DEF_RE = re.compile(r"^(triton_\w+)\s*=\s*async_compile\.triton\(")
_META_RE = re.compile(r"'num_load':\s*(\d+),\s*'num_store':\s*(\d+),\s*'num_reduction':\s*(\d+)")
_ASSERT_RE = re.compile(r"assert_size_stride\((\w+),\s*\(([\d, ]*)\),\s*\([\d, ]*\)(?:,\s*'input')?\)")
_EMPTY_RE = re.compile(r"(\w+)\s*=\s*empty_strided_\w+\(\(([\d, ]*)\),\s*\([\d, ]*\),\s*([\w.]+)\)")
_REINTERP_ASSIGN_RE = re.compile(r"(\w+)\s*=\s*reinterpret_tensor\((\w+),\s*\(([\d, ]*)\),")
_EXTERN_RE = re.compile(r"extern_kernels\.(\w+)\((.*)\)")
_RETURN_RE = re.compile(r"^\s*return\s*\((.*)\)\s*$")
_REINTERP_INLINE_RE = re.compile(r"reinterpret_tensor\((\w+),\s*\(([\d, ]*)\),")


def _split_args(argstr: str) -> list[str]:
    args, depth, cur = [], 0, []
    for ch in argstr:
        if ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        cur.append(ch)
    if cur:
        args.append("".join(cur).strip())
    return args


def _resolve_arg(arg: str, symtab: dict[str, TensorInfo]) -> TensorInfo | None:
    m = _REINTERP_INLINE_RE.match(arg)
    if m:
        base = symtab.get(m.group(1))
        return TensorInfo(_parse_shape(m.group(2)), base.dtype if base else "bf16")
    return symtab.get(arg)


def parse_subgraph(dump_dir: str) -> Subgraph:
    oc_path = os.path.join(dump_dir, "output_code.py")
    node_src, fx_tensors, gemm_nodes = parse_fx_readable(os.path.join(dump_dir, "fx_graph_readable.py"))
    sg = Subgraph(name=os.path.basename(dump_dir), path=dump_dir, node_source=node_src)
    gemm_queue = {op: [n for n, o in gemm_nodes if o == op] for op in {o for _, o in gemm_nodes}}

    lines = open(oc_path).read().splitlines()
    symtab: dict[str, TensorInfo] = dict(fx_tensors)  # primals dtypes come from the fx signature
    frag_inputs: dict[str, TensorInfo] = {}
    frag_nodes: dict[str, TensorInfo] = {}
    frag_ops: dict[str, str] = {}
    frag_returns: list[str] = []
    last_src: tuple[list[str], list[str]] | None = None
    pending_kernel: TritonKernel | None = None

    def flush_fragment() -> None:
        frag_inputs.clear(), frag_nodes.clear(), frag_ops.clear(), frag_returns.clear()

    for line in lines:
        s = _SRC_RE.search(line)
        if s:
            last_src = (
                [x.strip() for x in s.group(1).split(",") if x.strip()],
                [x.strip() for x in s.group(2).split(",") if x.strip()],
            )
            continue

        fn = _FRAG_NODE_RE.match(line)
        if fn:
            name, dt, shape = fn.group(1), fn.group(2), fn.group(3)
            info = TensorInfo(_parse_shape(shape), dt)
            if fn.group(4).startswith("PlaceHolder"):
                frag_inputs[name] = info
            else:
                frag_nodes[name] = info
                frag_ops[name] = fn.group(6).rsplit(".", 2)[-2] if fn.group(6).count(".") >= 2 else fn.group(6)
            continue
        fr = _FRAG_RET_RE.match(line)
        if fr and "%" in fr.group(1):
            frag_returns.extend(x.strip().lstrip("%") for x in fr.group(1).split(",") if x.strip())
            continue

        d = _DEF_RE.match(line)
        if d:
            k = TritonKernel(name=d.group(1))
            k.inputs = dict(frag_inputs)
            k.node_tensors = dict(frag_nodes)
            k.op_of_node = dict(frag_ops)
            for r in frag_returns:
                info = frag_nodes.get(r) or frag_inputs.get(r)
                if info is not None:
                    k.outputs[r] = info
            if last_src:
                k.src_nodes.update(last_src[0])
                k.aten_ops.update(a.replace("aten.", "") for a in last_src[1])
            for nd in list(k.src_nodes) + list(frag_nodes):
                if nd in node_src:
                    f, ln, code = node_src[nd]
                    k.source_lines[f"{f}:{ln}"] = code
            sg.triton[k.name] = k
            pending_kernel = k
            flush_fragment()
            continue
        if pending_kernel is not None:
            mt = _META_RE.search(line)
            if mt:
                pending_kernel.num_load, pending_kernel.num_store, pending_kernel.num_reduction = (
                    int(x) for x in mt.groups()
                )
                pending_kernel = None

        a = _ASSERT_RE.search(line)
        if a:
            name, shape = a.group(1), _parse_shape(a.group(2))
            prior = symtab.get(name)
            symtab[name] = TensorInfo(shape, prior.dtype if prior else "bf16")
            if "'input'" in line:
                sg.input_tensors[name] = symtab[name]
            continue
        e = _EMPTY_RE.search(line)
        if e:
            symtab[e.group(1)] = TensorInfo(_parse_shape(e.group(2)), e.group(3))
            continue
        r = _REINTERP_ASSIGN_RE.match(line.strip())
        if r:
            base = symtab.get(r.group(2))
            symtab[r.group(1)] = TensorInfo(_parse_shape(r.group(3)), base.dtype if base else "bf16")
            continue

        x = _EXTERN_RE.search(line)
        if x:
            call = ExternCall(op=x.group(1))
            if last_src:
                call.src_nodes.update(last_src[0])
                call.aten_ops.update(a.replace("aten.", "") for a in last_src[1])
            for arg in _split_args(x.group(2)):
                if arg.startswith("out="):
                    call.out = _resolve_arg(arg[4:], symtab)
                elif "=" in arg.split("(")[0]:
                    continue  # alpha=/beta= kwargs
                else:
                    call.args.append(_resolve_arg(arg, symtab))
            for nd in call.src_nodes:
                if nd in node_src:
                    f, ln, code = node_src[nd]
                    call.source_lines[f"{f}:{ln}"] = code
            # "Source Nodes" for externs use pre-AOT names that rarely resolve;
            # pair the Nth extern mm/addmm/... with the Nth such fx node instead.
            queue = gemm_queue.get(call.op)
            nd = queue.pop(0) if queue else None       # always pop to stay aligned
            if not call.source_lines and nd and nd in node_src:
                f, ln, code = node_src[nd]
                call.source_lines[f"{f}:{ln}"] = code
            sg.externs.append(call)
            continue

        ret = _RETURN_RE.match(line)
        if ret:
            for arg in _split_args(ret.group(1)):
                if not arg:
                    continue
                info = _resolve_arg(arg, symtab)
                # Only newly-materialized buffers count as boundary traffic;
                # returned primals are aliases of graph inputs (autograd saves).
                target = _REINTERP_INLINE_RE.match(arg).group(1) if arg.startswith("reinterpret_tensor") else arg
                if info is not None and not target.startswith(("primals", "arg", "tangents")):
                    sg.output_tensors.append(info)
    return sg


def parse_dump(trace_dir: str) -> list[Subgraph]:
    """Parse every compiled subgraph under an Inductor debug-dump directory."""
    dirs = sorted(os.path.dirname(p) for p in glob.glob(f"{trace_dir}/**/output_code.py", recursive=True))
    return [parse_subgraph(d) for d in dirs]
