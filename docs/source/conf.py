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
from sphinx.ext.autosummary import Autosummary
from sphinx.application import Sphinx
from unittest.mock import Mock

sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'RecTools'
copyright = '''
2022 MTS (Mobile Telesystems)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
author = 'MTS Big Data'

# The full version, including alpha/beta/rc tags
# release = '0.2.0'

# -- mock out modules
MOCK_MODULES = [
    'numpy',
    'pandas',
    'pandas.core',
    'pandas.core.dtypes',
    'pandas.core.dtypes.common',
    'scipy',
    'tqdm',
    'implicit',
    'implicit.als',
    'nmslib',
    'attrs',
    'typeguard',
    'lightfm',
    'torch',
    'torch.utils',
    'pytorch-lightning',
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'nbsphinx'
]

autodoc_typehints = "both"
autodoc_typehints_description_target = "all"
# add_module_names = False

# PACKAGES = [rectools.__name__]


# setup configuration
def skip(app, what, name, obj, skip, options):
    """
    Document __init__ methods
    """
    if name == "__init__":
        return True
    if name.startswith('_') and what in ('function', 'method'):
        return True
    return skip


def get_by_name(string: str):
    """
    Import by name and return imported module/function/class
    Args:
        string (str): module/function/class to import, e.g. 'pandas.read_csv' will return read_csv function as
        defined by pandas
    Returns:
        imported object
    """
    class_name = string.split(".")[-1]
    module_name = ".".join(string.split(".")[:-1])

    if module_name == "":
        return getattr(sys.modules[__name__], class_name)

    mod = __import__(module_name, fromlist=[class_name])
    return getattr(mod, class_name)


class ModuleAutoSummary(Autosummary):
    def get_items(self, names):
        new_names = []
        for name in names:
            mod = sys.modules[name]
            mod_items = getattr(mod, "__all__", mod.__dict__)
            for t in mod_items:
                if "." not in t and not t.startswith("_"):
                    obj = get_by_name(f"{name}.{t}")
                    if hasattr(obj, "__module__"):
                        mod_name = obj.__module__
                        t = f"{mod_name}.{t}"
                    if t.startswith("rectools"):
                        new_names.append(t)
        new_items = super().get_items(sorted(new_names, key=lambda x:  x.split(".")[-1]))
        return new_items


def setup(app: Sphinx):
    app.connect("autodoc-skip-member", skip)
    app.add_directive("moduleautosummary", ModuleAutoSummary)


autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '_templates']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'


html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
}

# html_context = {
#     'css_files': [
#         '_static/theme.css'
#     ],
# }

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = '_static/logo.jpeg'

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

