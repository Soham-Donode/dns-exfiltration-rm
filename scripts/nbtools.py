"""Small helpers shared by the notebook builders (plain JSON, no nbformat dependency)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _lines(source):
    lines = [line + "\n" for line in source.strip("\n").split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return lines


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source.strip())}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(source.strip())}


def embed(module):
    """A %%writefile cell carrying src/<module> so the notebook is self-contained on Colab."""
    return code(f"%%writefile {module}\n" + (ROOT / "src" / module).read_text(encoding="utf-8"))
