"""[all] is the union of the model extras (docs/installation.rst), and the
Docker images install Kokoro's spaCy tagger next to it."""

import re
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "mstar/integrations/dynamo/docker"
DOCKERFILES = [DOCKER_DIR / "Dockerfile", DOCKER_DIR / "Dockerfile.dynamo-base"]
# extras that are not a model family's runtime
NOT_MODEL_EXTRAS = {"all", "dev", "audio", "dynamo", "eval", "metrics", "omnivoice"}
# model-extra packages [all] leaves out, with the reason
NOT_IN_ALL = {
    # Intel XPU kernels; [all] targets CUDA
    "bagel_xpu": {"vllm-xpu-kernels"},
    # not in [all] yet: resemble-perth needs setuptools<82 next to it (#537)
    "chatterbox": {"librosa", "pyloudnorm", "resemble-perth", "setuptools", "tokenizers"},
    "chatterbox_multilingual": {"pykakasi", "spacy-pkuseg"},
}


def name(req):
    return re.sub(r"[-_.]+", "-", re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", req).group(0)).lower()


def version(text):
    return tuple(int(p) for p in re.findall(r"\d+", text))


def project():
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]


def mstar_install(path):
    """The `uv pip install ... -e ".[all...]"` RUN step, continuation lines joined."""
    text = path.read_text().replace("\\\n", " ")
    match = re.search(r'^RUN uv pip install [^\n]*-e "\.\[all[^\n]*$', text, re.MULTILINE)
    if match is None:
        raise AssertionError(f"{path.name}: no -e .[all] install step")
    return match.group(0)


class AllExtraTests(unittest.TestCase):
    def test_all_carries_every_model_extra(self):
        proj = project()
        extras = proj["optional-dependencies"]
        covered = {name(r) for r in extras["all"]} | {name(r) for r in proj["dependencies"]}
        for extra, reqs in extras.items():
            if extra in NOT_MODEL_EXTRAS:
                continue
            missing = {name(r) for r in reqs} - covered - NOT_IN_ALL.get(extra, set())
            self.assertEqual(missing, set(), f"[{extra}] packages missing from [all]")

    def test_kokoro_g2p_is_in_all(self):
        extras = project()["optional-dependencies"]
        self.assertLessEqual({name(r) for r in extras["kokoro"]}, {name(r) for r in extras["all"]})


class DockerSpacyTaggerTests(unittest.TestCase):
    def test_images_install_the_tagger_with_the_all_extra(self):
        kokoro = project()["optional-dependencies"]["kokoro"]
        spacy_floor = version(next(r for r in kokoro if name(r) == "spacy").split(">=")[1])
        for path in DOCKERFILES:
            pin = re.search(r"^ARG EN_CORE_WEB_SM_VERSION=(\S+)$", path.read_text(), re.MULTILINE)
            self.assertIsNotNone(pin, path.name)
            self.assertGreaterEqual(version(pin.group(1))[:2], spacy_floor[:2], path.name)
            # one resolve: the tagger and the spaCy minor it was built for, with [all]
            step = mstar_install(path)
            self.assertIn('"spacy~=${EN_CORE_WEB_SM_VERSION}"', step, path.name)
            self.assertIn(
                '"en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/'
                'en_core_web_sm-${EN_CORE_WEB_SM_VERSION}/en_core_web_sm-${EN_CORE_WEB_SM_VERSION}-py3-none-any.whl"',
                step,
                path.name,
            )


if __name__ == "__main__":
    unittest.main()
