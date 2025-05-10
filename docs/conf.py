import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'pyco2stats'
author = 'Maurizio Petrelli, Alessandra Ariano, Marco Baroni'
release = '0.1.0'

# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
'sphinx.ext.mathjax',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# List of patterns, relative to source directory, that match files and directories to ignore when looking for source files.
exclude_patterns = []
