# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import inspect

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "PyVBMC"
copyright = "2021, Machine and Human Intelligence research group (PI: Luigi Acerbi, University of Helsinki)"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.linkcode",
    "myst_nb",
]
numpydoc_show_class_members = False

autodoc_default_options = {
    "members": "var1, var2",
    "special-members": "__call__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

def linkcode_resolve(domain, info):
    """
    Used for sphinx.ext.linkcode.
    Modified from the basic example from the extension.
    """
    if domain != "py" or not info["module"]:
        return None

    obj = sys.modules[info["module"]]
    for part in info["fullname"].split("."):
        obj = getattr(obj, part)

    # unwrap to get rid of decorators.
    filename = inspect.getsourcefile(inspect.unwrap(obj))
    
    # to get rid of the local path, quiet hacky, but works
    filename = filename[filename.index("pyvbmc/") + 7 :]
    return "https://github.com/lacerbi/pyvbmc/tree/main/%s" % filename


coverage_show_missing_items = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_title = "PyVBMC"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["css/custom.css"]

todo_include_todos = True

jupyter_execute_notebooks = "off"