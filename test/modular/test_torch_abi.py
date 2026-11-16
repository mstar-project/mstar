"""The Rust torch wrapper against Python's view of the same tensors.

Run across torch versions by loading a shim built elsewhere: set
``MSTAR_TEST_TORCH_ABI_SHIM`` to its path (the version matrix in the PR)."""
import os
import re
import tomllib
from pathlib import Path

import pytest
import torch

rs = pytest.importorskip("mstar_rust")

if not hasattr(rs, "torch_abi_load"):
    pytest.skip("this mstar_rust predates the torch wrapper", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    shim = os.environ.get("MSTAR_TEST_TORCH_ABI_SHIM")
    if shim is None and not rs.torch_abi_embedded():
        pytest.skip("this mstar_rust was built without a shim")
    print(rs.torch_abi_load(torch.__version__, shim))


def _devices() -> list[str]:
    return ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]


DTYPES = [
    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.bfloat16,
    torch.float32, torch.float64, torch.bool, torch.complex64,
]


def _expect_facts(t: torch.Tensor) -> None:
    f = rs.torch_abi_facts(t)
    assert f["plain"] is True
    assert f["data"] == t.data_ptr()
    assert f["nbytes"] == t.numel() * t.element_size()
    assert f["device_type"] == (1 if t.is_cuda else 0)
    assert f["device_index"] == (t.device.index if t.is_cuda else -1)
    assert f["dtype"] == rs.torch_abi_dtype(t.dtype)
    assert f["dims"] == list(t.shape) and f["ndim"] == t.dim()
    assert f["contiguous"] is t.is_contiguous()


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", DTYPES)
def test_facts_match_python_for_every_dtype(device, dtype):
    _expect_facts(torch.zeros(2, 3, dtype=dtype, device=device))


@pytest.mark.parametrize("device", _devices())
def test_facts_match_python_for_every_shape_and_view(device):
    base = torch.arange(120, dtype=torch.float32, device=device).reshape(2, 3, 4, 5)
    for t in (
        base, base[1:], base[:, 1], base[..., 2:4], base.permute(3, 0, 1, 2), base[0, 0, 0, 0],
        base.reshape(-1), base[:0], torch.empty(0, 3, device=device), base.detach(),
        torch.tensor(1.5, device=device), torch.nn.Parameter(base.clone()),
        base.transpose(0, 1), base[::2],
    ):
        _expect_facts(t)


def test_a_non_tensor_is_refused_rather_than_read():
    for not_a_tensor in (None, 3, "x", [1, 2], torch.float32):
        with pytest.raises(TypeError, match="torch.Tensor"):
            rs.torch_abi_facts(not_a_tensor)


def test_a_tensor_whose_bytes_are_not_its_values_is_not_plain():
    z = torch.zeros(4, dtype=torch.complex64)
    for t in (z.conj(), torch.zeros(4, dtype=torch.complex64).conj().imag,
              torch.eye(3).to_sparse()):
        assert rs.torch_abi_facts(t)["plain"] is False


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", DTYPES)
def test_empty_makes_a_real_tensor(device, dtype):
    d = torch.device(device)
    t = rs.torch_abi_empty([3, 0, 5] if dtype == torch.bool else [3, 5], dtype,
                           1 if d.type == "cuda" else 0, d.index if d.index is not None else -1)
    assert type(t) is torch.Tensor and t.device == d and t.dtype == dtype and not t.requires_grad
    assert t.grad_fn is None
    _expect_facts(t)
    t.zero_()
    assert not t.bool().any()


def test_empty_honours_inference_mode_as_torch_empty_does():
    with torch.inference_mode():
        assert rs.torch_abi_empty([2], torch.float32).is_inference()
    assert not rs.torch_abi_empty([2], torch.float32).is_inference()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for pinned host memory")
def test_empty_pins_on_request():
    assert rs.torch_abi_empty([64], torch.uint8, pinned=True).is_pinned()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_an_allocation_too_large_raises_torchs_oom():
    with pytest.raises(torch.cuda.OutOfMemoryError):
        rs.torch_abi_empty([1 << 50], torch.uint8, 1, 0)


@pytest.mark.parametrize("device", _devices())
def test_narrow_is_the_view_torch_gives(device):
    u = torch.arange(60.0, device=device).reshape(5, 12)
    s = rs.torch_abi_narrow(u, 1, 2)
    assert torch.equal(s, u[1:3]) and s.data_ptr() == u[1:3].data_ptr()
    assert s._base is u or s._base is u._base
    with pytest.raises(RuntimeError):
        rs.torch_abi_narrow(u, 4, 9)


@pytest.mark.parametrize("device", _devices())
def test_lead_contiguous_matches_permute(device):
    u = torch.arange(120.0, device=device).reshape(2, 3, 4, 5)
    for dim in range(4):
        order = [dim, *(i for i in range(4) if i != dim)]
        led = rs.torch_abi_lead_contiguous(u, dim)
        assert led.is_contiguous() and torch.equal(led, u.permute(order))
    with pytest.raises(RuntimeError):
        rs.torch_abi_lead_contiguous(u, 9)


@pytest.mark.parametrize("device", _devices())
def test_cat_matches_torch(device):
    u = torch.arange(60.0, device=device).reshape(5, 12)
    assert torch.equal(rs.torch_abi_cat([u, u, u], 0), torch.cat([u, u, u], 0))
    assert torch.equal(rs.torch_abi_cat([u, u], 1), torch.cat([u, u], 1))
    with pytest.raises(RuntimeError):
        rs.torch_abi_cat([u, u[0]], 0)
    with pytest.raises(TypeError):
        rs.torch_abi_cat([u, None], 0)


def test_the_startup_check_passes_here():
    from mstar.communication.torch_abi import require_torch_abi

    for device in _devices():
        assert require_torch_abi(device)


@pytest.mark.parametrize("version", ["2.9.1", "2.13.0", "3.0.0", "nightly"])
def test_an_unsupported_torch_is_refused_saying_what_to_add(version):
    for refuse in (rs.torch_abi_check_version, rs.torch_abi_load):
        with pytest.raises(RuntimeError, match=r"TorchV2 covers torch>=2\.10,<2\.13.*implementor"):
            refuse(version)


def test_the_supported_range_is_the_one_pyproject_pins():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("no pyproject.toml beside this test (a copied test tree)")
    deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    pin = next(d for d in deps if re.match(r"torch\b(?!vision|audio)", d))
    match = re.fullmatch(r"torch>=(\d+)\.(\d+)(?:\.0)?,<(\d+)\.(\d+)(?:\.0)?", pin)
    assert match, pin
    lo, hi = f"{match[1]}.{match[2]}", f"{match[3]}.{match[4]}"
    with pytest.raises(RuntimeError) as refused:
        rs.torch_abi_check_version("0.0.0")
    assert f"torch>={lo},<{hi}" in str(refused.value), f"pyproject pins {pin}"
