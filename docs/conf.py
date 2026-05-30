import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

try:
    from mminf import __version__ as package_version
except Exception:  # noqa: BLE001
    package_version = "0.0.0"

project = "mminf"
author = "mminf Team"
current_year = datetime.now().year
copyright = f"2025-{current_year}, {author}"
version = package_version
release = package_version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "sphinx.ext.mathjax",
]

autosectionlabel_prefix_document = True
autodoc_typehints = "description"

myst_enable_extensions = [
    "dollarmath",
    "deflist",
    "colon_fence",
    "linkify",
]
myst_heading_anchors = 3

templates_path = ["_templates"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = f"{project} v{version} Documentation"
html_copy_source = True
html_last_updated_fmt = ""

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

html_theme_options = {
    "repository_url": "https://github.com/merceod/multimodal_inference",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "show_navbar_depth": 2,
    "collapse_navbar": False,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_source_button": True,
    "show_toc_level": 2,
}

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
autodoc_preserve_defaults = True
navigation_with_keys = False

# Heavy / optional native deps that need not be importable to build the docs.
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "torchcodec",
    "transformers",
    "triton",
    "flashinfer",
    "huggingface_hub",
    "numpy",
    "zmq",
    "safetensors",
    "einops",
    "yaml",
]
