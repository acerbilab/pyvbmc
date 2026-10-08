# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "PyVBMC"
copyright = "2022, Machine and Human Intelligence research group (PI: Luigi Acerbi, University of Helsinki)"


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
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.extlinks",
]

numpydoc_show_class_members = False
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ["http", "https", "mailto"]
autodoc_default_options = {
    "members": "var1, var2",
    "special-members": "__call__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Define shorthand for external links:
extlinks = {
    "labrepos": ("https://github.com/acerbilab/%s", None),
    "mainbranch": ("https://github.com/acerbilab/pyvbmc/blob/main/%s", None),
}

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
html_static_path = ["css/wrap.css"]
html_css_files = ["wrap.css"]
html_show_sourcelink = False
html_theme_options = {
    "extra_footer": (
        '<p>More <a href="https://acerbilab.org/model-fitting/">model-fitting '
        "tools</a> from "
        '<a href="https://www.helsinki.fi/en/researchgroups/'
        "machine-and-human-intelligence\">Luigi Acerbi's group</a>.</p>"
    ),
    "repository_url": "https://github.com/acerbilab/pyvbmc",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
        "colab_url": "https://colab.research.google.com/",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "logo": {"alt-text": "PyVBMC"},
    "path_to_docs": "docsrc/source",
}
html_logo = "../../logo.svg"
html_baseurl = "https://acerbilab.github.io/pyvbmc/"
html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"
]

todo_include_todos = True

# do not execute jupyter notebooks when building docs
nb_execution_mode = "off"

# download notebooks as .ipynb and not as .ipynb.txt
html_sourcelink_suffix = ""

suppress_warnings = [
    f"autosectionlabel._examples/{filename.split('.')[0]}"
    for filename in os.listdir("../../examples")
    if os.path.isfile(os.path.join("../../examples", filename))
]  # Avoid duplicate label warnings for Jupyter notebooks.
# Example 2 stores its Plotly figure twice: as HTML, which the page shows, and
# as Plotly JSON for notebook viewers, which myst-nb cannot render and warns
# about. The suppression covers any output type myst-nb cannot render.
suppress_warnings.append("mystnb.unknown_mime_type")


def notebook_source_links(app, pagename, templatename, context, doctree):
    """Point notebook launch and edit buttons to their repository sources."""
    if not pagename.startswith("_examples/"):
        return
    source = pagename + context["page_source_suffix"]
    staged = app.config.html_theme_options["path_to_docs"] + "/" + source
    original = source.replace("_examples/", "examples/", 1)
    for group in context.get("header_buttons", []):
        for button in group.get("buttons", [group]):
            if "url" in button:
                button["url"] = button["url"].replace(staged, original)


def setup(app):
    # The theme builds its buttons at priority 501. The notebooks are
    # copied into source/_examples for rendering but live in examples/.
    app.connect("html-page-context", notebook_source_links, priority=900)
