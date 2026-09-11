from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inside(root: Path, value: str, *, exists: bool = True) -> Path:
    root = root.resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes the workspace: {value}")
    if exists and not path.exists():
        raise ValueError(f"Path is missing: {value}")
    return path


class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve()
        marker = self.root / ".btl-workspace.json"
        layout = read_json(marker) if marker.is_file() else {}
        if layout and layout.get("schema_version") != 1:
            raise ValueError("Unsupported workspace layout version")
        self.lab = inside(self.root, layout.get("lab", "btl-lab"))
        self.registry = inside(self.root, layout.get("registry", "btl-lab/projects.json"))
        self.state = inside(self.root, layout.get("state", "btl-lab/.state"), exists=False)
        self.catalog = self.lab / "recipes.json"
        if not self.registry.is_file():
            raise ValueError(f"No BTL registry at {self.registry}; pass --workspace")

    def projects(self) -> list[dict]:
        document = read_json(self.registry)
        if document.get("schema_version") != 1:
            raise ValueError("Unsupported project registry version")
        projects = document["projects"]
        ids = [project["id"] for project in projects]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate project IDs in registry")
        return projects

    def project(self, project_id: str) -> dict:
        for project in self.projects():
            if project["id"] == project_id:
                return project
        raise ValueError(f"Unknown project: {project_id}")

    def recipes(self) -> list[dict]:
        document = read_json(self.catalog)
        if document.get("schema_version") != 1:
            raise ValueError("Unsupported recipe catalog version")
        recipes = document["recipes"]
        ids = [recipe["id"] for recipe in recipes]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate recipe IDs")
        return recipes

    def recipe(self, recipe_id: str) -> dict:
        for recipe in self.recipes():
            if recipe["id"] == recipe_id:
                self.project(recipe["project_id"])
                return recipe
        raise ValueError(f"Unknown recipe: {recipe_id}")
