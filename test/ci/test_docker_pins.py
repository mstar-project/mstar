"""Keep the two Dockerfiles' pins consistent with each other and with pyproject.toml."""

import re
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "mstar/integrations/dynamo/docker"
DOCKERFILES = [DOCKER_DIR / "Dockerfile", DOCKER_DIR / "Dockerfile.dynamo-base"]
PIN_ARGS = ("TORCH_INDEX", "TORCH_VERSION", "TORCHVISION_VERSION", "TORCHAUDIO_VERSION",
            "TORCHCODEC_VERSION", "FLASH_ATTN_WHEEL", "MOONCAKE_PACKAGE")
# Mooncake's wheel per CUDA major; the extras name the CUDA 12 one (#480)
MOONCAKE_WHEELS = {"12": "mooncake-transfer-engine", "13": "mooncake-transfer-engine-cuda13"}


def version(text):
    return tuple(int(p) for p in re.findall(r"\d+", text))


def build_args(path):
    return dict(re.findall(r"^ARG (\w+)=(\S+)$", path.read_text(), re.MULTILINE))


def mstar_install(path):
    """The `uv pip install ... -e ".[all...]"` RUN step, continuation lines joined."""
    text = path.read_text().replace("\\\n", " ")
    match = re.search(r'^RUN uv pip install [^\n]*-e "\.\[all[^\n]*$', text, re.MULTILINE)
    if match is None:
        raise AssertionError(f"{path.name}: no -e .[all] install step")
    return match.group(0)


def cuda_major(args):
    return re.search(r"cu(\d+?)\d$", args["TORCH_INDEX"]).group(1)


class DockerPinTests(unittest.TestCase):
    def test_both_dockerfiles_pin_the_same_torch_stack(self):
        first, second = (build_args(p) for p in DOCKERFILES)
        for name in PIN_ARGS:
            self.assertIn(name, first)
            self.assertEqual(first[name], second.get(name), name)

    def test_torch_stack_matches_the_pyproject_floor_and_itself(self):
        args = build_args(DOCKERFILES[0])
        deps = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
        floor = next(d for d in deps if d.startswith("torch>="))
        self.assertGreaterEqual(version(args["TORCH_VERSION"]), version(floor.split(">=")[1].split(",")[0]))
        torch_minor = version(args["TORCH_VERSION"])[1]
        # torchcodec 0.N pairs with torch 2.N
        self.assertEqual(version(args["TORCHCODEC_VERSION"])[1], torch_minor)
        self.assertIn(f"+cu{cuda_major(args)}torch2.{torch_minor}cxx11", args["FLASH_ATTN_WHEEL"])

    def test_runtime_setuptools_meets_the_build_floor(self):
        # torch pulls setuptools from the torch index, whose copy is older than the build floor
        requires = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["build-system"]["requires"]
        build_floor = version(next(r for r in requires if r.startswith("setuptools>=")).split(">=")[1])
        for path in DOCKERFILES:
            pin = re.search(r'"setuptools>=([\d.]+)"', mstar_install(path))
            self.assertIsNotNone(pin, f"{path.name}: no setuptools floor in the .[all] install")
            self.assertGreaterEqual(version(pin.group(1)), build_floor, path.name)

    def test_mooncake_wheel_matches_the_cuda_major(self):
        for path in DOCKERFILES:
            args = build_args(path)
            self.assertEqual(args["MOONCAKE_PACKAGE"], MOONCAKE_WHEELS[cuda_major(args)], path.name)
            text = path.read_text().replace("\\\n", " ")
            install = text.index(mstar_install(path))
            swap = re.search(r'^RUN if \[ "\$\{MOONCAKE_PACKAGE\}" != mooncake-transfer-engine \][^\n]*$',
                             text, re.MULTILINE)
            self.assertIsNotNone(swap, f"{path.name}: no Mooncake swap step")
            self.assertGreater(swap.start(), install, f"{path.name}: the swap must follow the .[all] install")
            step = swap.group(0)
            self.assertIn("uv pip uninstall --python /opt/mstar/venv/bin/python mooncake-transfer-engine", step)
            # same version as the extras resolved, without letting the resolver near the torch pins
            self.assertIn('--no-deps "${MOONCAKE_PACKAGE}==${v}"', step)
            self.assertIn("m.version('mooncake-transfer-engine')", step)

    def test_the_cuda_base_image_gets_mooncakes_system_libraries(self):
        # Mooncake's engine links libibverbs, libmlx5 and libcurl; the CUDA -devel base has none
        # of them (the Dynamo runtime base ships them)
        text = DOCKERFILES[0].read_text().replace("\\\n", " ")
        apt = re.search(r"^RUN apt-get update && apt-get install [^\n]*$", text, re.MULTILINE).group(0)
        for package in ("libibverbs1", "ibverbs-providers", "libcurl4"):
            self.assertIn(package, apt.split())

    def test_the_cuda_12_override_names_every_cuda_dependent_arg(self):
        header = "\n".join(line for line in DOCKERFILES[0].read_text().splitlines() if line.startswith("#"))
        overrides = dict(re.findall(r"--build-arg (\w+)=(\S+)", header))
        self.assertTrue(overrides["TORCH_INDEX"].endswith("/cu128"))
        self.assertEqual(overrides["MOONCAKE_PACKAGE"], MOONCAKE_WHEELS["12"])
        self.assertIn("CUDA_IMAGE", overrides)
        self.assertIn("FLASH_ATTN_WHEEL", overrides)


if __name__ == "__main__":
    unittest.main()
