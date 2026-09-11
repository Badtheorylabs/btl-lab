import json
import pytest
from btl_lab.workspace import Workspace, sha256
from btl_lab.store import Store


@pytest.fixture
def lab(tmp_path):
    folder = tmp_path / "btl-lab"
    folder.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("source snapshot\n")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": [
        {"path": "source.txt", "sha256": sha256(source)}
    ]}))
    (folder / "projects.json").write_text(json.dumps({"schema_version": 1, "projects": [
        {"id": "lab", "path": "btl-lab", "source_documents": ["source.txt"],
         "primary_program": "lab", "lifecycle": "untriaged", "owner": None}
    ]}))
    (folder / "recipes.json").write_text(json.dumps({"schema_version": 1, "recipes": [
        {"id": "registry", "project_id": "lab", "backend": "local-audit", "operation": "registry-audit"},
        {"id": "source", "project_id": "lab", "backend": "local-audit", "operation": "manifest-audit",
         "manifest": "manifest.json"},
        {"id": "rl", "project_id": "lab", "backend": "prime-rl", "operation": "rl",
         "revision": "a" * 40, "config": "rl.toml", "required_gpus": 2}
    ]}))
    workspace = Workspace(tmp_path)
    store = Store(folder / ".state/lab.sqlite3")
    yield workspace, store
    store.db.close()
