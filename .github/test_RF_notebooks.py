"""
Tests for checking that all Jupyter notebooks in repository execute without errors.
Based on https://github.com/mcullan/jupyter-actions/blob/master/test/test_notebooks.py
Leif Denby - MIT License 2021
"""
import os
import subprocess
import pytest
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(TEST_DIR, "../")


def _find_notebooks():
    # Get all files included in the git repository
    git_files = (
        subprocess.check_output(
            "git ls-tree --full-tree --name-only -r HEAD", shell=True
        )
        .decode("utf-8")
        .splitlines()
    )

    # Get just the notebooks from the git files
    notebooks_files = [Path(fn) for fn in git_files if fn.endswith("RandomForests.ipynb")]

    # remove all notebooks that haven't been checked
    notebooks_files = [p for p in notebooks_files if p.parent.name != "unchecked"]
    return [str(p) for p in notebooks_files]


@pytest.mark.parametrize("notebook_filename", _find_notebooks())
def test_notebook(notebook_filename, html_directory="notebook-html"):
    """
    Checks if an IPython notebook runs without error from start to finish.
    """
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600)

    # Check that the notebook runs
    ep.preprocess(nb, {"metadata": {"path": Path(notebook_filename).parent }})
