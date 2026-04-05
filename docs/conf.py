import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "bartorch"
author = "bartorch contributors"
version = "0.1.0"
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/mcencini/bartpy",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs",
    "show_navbar_depth": 2,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# Notebooks live in the top-level examples/ directory (symlinked as docs/examples/).
# Never execute notebooks during the Sphinx build — they require the compiled C++
# extension and are provided for reading / download only.
nbsphinx_execute = "never"

# Follow symlinks so that docs/examples/ → ../examples/ is resolved correctly.
# This is the default in Sphinx ≥ 7 but we set it explicitly for clarity.
html_extra_path = []

autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

templates_path = ["_templates"]
html_static_path = ["_static"]
