from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

install_requires = [
"faiss-cpu",
"PyYAML",
"pandas",
"scikit-learn",
"numpy",
"h5py",
"tqdm"
]

setup_requires = []

# extras_require = {"hyperopt": ["hyperopt==0.2.5"], "ray": ["ray>=1.13.0"]}
extras_require = {}
classifiers = ["License :: OSI Approved :: MIT License"]

long_description = (
    "matchbox is developed based on Python and PyTorch for "
    "reproducing and developing recommendation algorithms in "
    "a unified, comprehensive and efficient framework for "
    "research purpose. In the first version, our library "
    "includes 53 recommendation algorithms, covering four "
    "major categories: General Recommendation, Sequential "
    "Recommendation, Context-aware Recommendation and "
    "Knowledge-based Recommendation. View matchbox homepage "
    "for more information: https://matchbox.io"
)

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:
    install_requires.extend(setup_requires)

setup(
    name="matchbox",
    version="1.1.1",  # please remember to edit matchbox/__init__.py in response, once updating the version
    description="A unified, comprehensive and efficient recommendation library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xue-pai/ModelBox.git",
    author="matchboxTeam",
    author_email="matchbox@outlook.com",
    packages=[package for package in find_packages() if package.startswith("matchbox")],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False,
    classifiers=classifiers,
)

