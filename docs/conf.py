"""Sphinx configuration for the actgpr API documentation."""

from importlib.metadata import version as get_version

# -- Project information -----------------------------------------------------

project = "actgpr"
author = "Alexander Schroder"
copyright = "2026, Alexander Schroder"
release = get_version("actgpr")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",  # pull in docstrings
    "sphinx.ext.napoleon",  # parse NumPy-style docstrings
    "sphinx.ext.viewcode",  # link to highlighted source
    "sphinx.ext.intersphinx",  # cross-reference torch/numpy docs
]

templates_path = ["_templates"]
exclude_patterns = ["build", "project.md"]

# NumPy-style docstrings only
napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = []
