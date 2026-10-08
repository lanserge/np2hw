# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The toy library beside these tests is found as any library is: by the
import path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import itertools
import json

import pytest

_count = itertools.count()


@pytest.fixture
def library(tmp_path, monkeypatch):
    """Write {module: {block: {...}}} as a package; return a function that
    takes the files and gives back the package name."""
    monkeypatch.syspath_prepend(str(tmp_path))

    def make(files, python: dict | None = None) -> str:
        """``files`` is {module: {block: {...}}}, or a function of the
        package name returning it, so a file can reference its own library."""
        name = f"fixture_lib_{next(_count)}"
        package = tmp_path / name
        package.mkdir()
        (package / "__init__.py").write_text("")
        if callable(files):
            files = files(name)
        for module, content in files.items():
            # {block: {...}}, or a whole library file (blocks, profiles)
            whole = set(content) & {"blocks", "profiles"} == set(content) \
                and content
            (package / f"{module}.json").write_text(
                json.dumps(content if whole else {"blocks": content}))
        for module, text in (python or {}).items():
            (package / f"{module}.py").write_text(text)
        return name
    return make

