from __future__ import annotations

from pathlib import Path

from .workspace import Workspace, inside, sha256
from btl_kernels.manifest import audit_manifest


def registry_audit(workspace: Workspace) -> dict:
    checks = []
    projects = workspace.projects()
    for project in projects:
        for value in [project["path"], *project.get("source_documents", [])]:
            try:
                inside(workspace.root, value)
                checks.append({"path": value, "passed": True})
            except ValueError as error:
                checks.append({"path": value, "passed": False, "error": str(error)})
    return {"passed": all(c["passed"] for c in checks), "projects": len(projects),
            "checks": checks, "scope": "Registry paths only; no project readiness or model evaluation"}


def manifest_audit(workspace: Workspace, relative: str) -> dict:
    return audit_manifest(workspace.root, relative)

def artifact_record(workspace: Workspace, relative: str, role: str, evidence_status: str) -> dict:
    path = inside(workspace.root, relative)
    if not path.is_file():
        raise ValueError("Attach a file or a manifest, not a directory")
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise ValueError("Artifact changed while hashing")
    return {"path": str(path.relative_to(workspace.root)), "sha256": digest,
            "bytes": after.st_size, "role": role, "evidence_status": evidence_status}


def verify_artifacts(workspace: Workspace, run: dict) -> dict:
    checks = []
    for artifact in run["artifacts"]:
        try:
            path = inside(workspace.root, artifact["path"])
            passed = path.stat().st_size == artifact["bytes"] and sha256(path) == artifact["sha256"]
            checks.append({"path": artifact["path"], "passed": passed})
        except (ValueError, OSError) as error:
            checks.append({"path": artifact["path"], "passed": False, "error": str(error)})
    return {"passed": bool(checks) and all(c["passed"] for c in checks), "checks": checks,
            "scope": "Recorded artifact integrity, not independent verification of reported claims"}
