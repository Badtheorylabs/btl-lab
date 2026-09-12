"""Create portable workspace metadata, without downloads or model execution."""
from __future__ import annotations

import json
import os
import tempfile
from importlib.resources import files
from pathlib import Path

from .workspace import Workspace


def initialize(root: Path) -> dict:
    root = root.expanduser().resolve()
    marker = root / ".btl-workspace.json"
    if marker.exists():
        workspace = Workspace(root)
        workspace.projects()
        workspace.recipes()
        return {"workspace": str(root), "created": False, "status": "already-initialized"}
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("Initialize a new or empty directory; existing files will not be overwritten")
    resources = files("btl_lab").joinpath("resources")
    names = ("recipes.json", "backends.json", "projects.json", "baseline.json", "candidate.json", "study.json")
    content = {name: resources.joinpath(name).read_text() for name in names}
    layout = {"schema_version": 1, "lab": ".btl", "registry": ".btl/projects.json",
              "state": ".btl/state", "groups": ["examples"]}
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".btl-init-", dir=root.parent) as temporary:
        stage = Path(temporary) / "workspace"
        (stage / ".btl").mkdir(parents=True)
        (stage / "examples").mkdir()
        for index, name in enumerate(names):
            folder = ".btl" if index < 3 else "examples"
            (stage / folder / name).write_text(content[name], encoding="utf-8")
        (stage / ".btl/.gitignore").write_text("state/\n", encoding="utf-8")
        (stage / ".btl-workspace.json").write_text(json.dumps(layout, indent=2) + "\n")
        Workspace(stage).projects()
        os.replace(stage, root)
    Workspace(root).projects()
    return {"workspace": str(root), "created": True, "status": "initialized",
            "next_commands": [f'btl --workspace "{root}" doctor',
                              f'btl --workspace "{root}" run registry-audit',
                              f'btl --workspace "{root}" measure compare examples/baseline.json examples/candidate.json'],
            "scope": "Workspace metadata and illustrative examples; no model execution"}
