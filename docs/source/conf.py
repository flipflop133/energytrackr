"""Configuration file for the Sphinx documentation builder."""

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "energytrackr"
copyright = "2025, François Bechet"
author = "François Bechet"
release = "0.1.0"


extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]
exclude_patterns = []
autosummary_generate = True
html_theme = "furo"
autodoc_mock_imports = []
