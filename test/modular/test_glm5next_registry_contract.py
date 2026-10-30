"""The registry names glm5_next lazily, and the model module imports under the test stubs."""
import pytest


def _import_or_skip(module_name: str):
    """Import; skip on a missing third-party module, fail on a missing mstar one."""
    try:
        return __import__(module_name, fromlist=["_"])
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] == "mstar":
            raise
        pytest.skip(f"{module_name} needs {e.name!r} (box venvs have it)")


def test_full_model_module_imports_under_stubs():
    glm = _import_or_skip("mstar.model.glm5_next.glm5_next_model")
    assert hasattr(glm, "Glm5NextModel")


def test_registry_entry_resolves_lazily():
    from mstar.model import registry

    # The lazy tuple form: the registry names the module and class without
    # importing them, so listing models never pulls a model's deps.
    assert registry.MODEL_REGISTRY["glm5_next"] == (
        "mstar.model.glm5_next.glm5_next_model", "Glm5NextModel",
    )
    assert registry.HF_MODELS["glm5_next"]["model_path_hf"] == "zai-org/GLM-5.3-Flash"

    glm = _import_or_skip("mstar.model.glm5_next.glm5_next_model")
    assert registry.get_model_class("glm5_next") is glm.Glm5NextModel
