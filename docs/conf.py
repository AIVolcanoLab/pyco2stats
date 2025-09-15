import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'pyco2stats'
author = 'Maurizio Petrelli, Alessandra Ariano, Marco Baroni, Frondini Francesco, Ágreda-López Mónica, Chiodini Giovanni'
release = '0.1.9'

# Add any Sphinx extension module names here, as strings.
extensions = [
    'nbsphinx',                   # for notebooks
    'sphinx.ext.autodoc',        # Auto-document docstrings
    'sphinx.ext.napoleon',       # Support for Google/NumPy-style docstrings
    'sphinx.ext.intersphinx',    # Link to other project's documentation
    'sphinx.ext.viewcode',       # Add links to highlighted source code
    'sphinx.ext.mathjax',        # LaTeX math rendering
    'sphinx.ext.autosummary',    # Generate function/class summary tables
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# List of patterns, relative to source directory, that match files and directories to ignore when looking for source files.
exclude_patterns = []

html_static_path = ['_static']
