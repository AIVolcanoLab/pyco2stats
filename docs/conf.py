from pathlib import Path
import sys
import tomllib


# Repository root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Project information
project = "pyco2stats"
author = (
    "Maurizio Petrelli, Marco Baroni, Alessandra Ariano, Monica Ágreda-López, Lisa Ricci, Francesco Frondini, Giuseppe Saldi, Carlo Cardellini, Diego Perugini, Giovanni Chiodini"
)

# Read the version directly from pyproject.toml
with (ROOT / "pyproject.toml").open("rb") as file:
    pyproject = tomllib.load(file)

release = pyproject["project"]["version"]
version = release


extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme",
]


# Autodoc configuration
autodoc_member_order = "bysource"
autosummary_generate = True

# Your project currently contains both NumPy-style and Google-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = True


templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
]


html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}


# Enables the GitHub source links supported by sphinx_rtd_theme
html_context = {
    "display_github": True,
    "github_user": "AIVolcanoLab",
    "github_repo": "pyco2stats",
    "github_version": "main",
    "conf_py_path": "/docs/",
}


html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
