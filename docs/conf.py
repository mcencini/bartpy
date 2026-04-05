import os
import sys

# Repository root is one level above docs/.  The package lives under src/ in the
# modern src-layout; add that so Sphinx autodoc can import bartorch directly.
sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath(".."))

project = "bartorch"
author = "bartorch contributors"
version = "0.1.0"
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
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

# Enable intersphinx cross-references when internet access is available
# (ReadTheDocs builds).  Disable in offline CI environments to avoid network
# warnings that would break a -W sphinx-build invocation.
_online = os.environ.get("READTHEDOCS") == "True"
intersphinx_mapping = (
    {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable", None),
        "torch": ("https://pytorch.org/docs/stable", None),
    }
    if _online
    else {}
)

# Notebooks live in the top-level examples/ directory (symlinked as docs/examples/).
# Always execute notebooks so the rendered docs show cell outputs (including error
# tracebacks from stub cells while the C++ extension is not yet wired up).
nbsphinx_execute = "always"
# Allow cells that raise errors (e.g. stub RuntimeError from run()) without
# failing the Sphinx build.  Tracebacks are rendered inline in the docs.
# TODO: remove once _bartorch_ext.run() is fully implemented (Phase 1 roadmap).
nbsphinx_allow_errors = True

# Suppress Pygments highlight-failure warnings from raw-markdown cells that
# contain fenced code blocks (the backtick syntax confuses the IPython lexer).
# Also suppress image.not_readable: when stub cells raise mid-way through
# a matplotlib figure creation, inline images are captured but their backing
# files may not be materialised at the expected nbsphinx doctrees path.
suppress_warnings = [
    "misc.highlighting_failure",
    "image.not_readable",
]

# Follow symlinks so that docs/examples/ → ../examples/ is resolved correctly.
# This is the default in Sphinx ≥ 7 but we set it explicitly for clarity.
html_extra_path = []

autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

templates_path = ["_templates"]
html_static_path = ["_static"]
