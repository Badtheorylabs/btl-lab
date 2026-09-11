from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    """Small transactional ledger. Imported evidence never changes run outcome."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=15)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
                recipe_id TEXT NOT NULL, kind TEXT NOT NULL,
                status TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, plan TEXT NOT NULL, result TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                created_at TEXT NOT NULL, status TEXT NOT NULL, detail TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                path TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL,
                role TEXT NOT NULL, evidence_status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS project_overrides (
                project_id TEXT PRIMARY KEY, fields TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY, project_id TEXT NOT NULL,
                run_id TEXT REFERENCES runs(id), text TEXT NOT NULL, created_at TEXT NOT NULL
            );
        """)

    def create(self, plan: dict, *, kind: str = "check") -> str:
        with self.db:
            return self.create_in_transaction(plan, kind=kind)

    def create_in_transaction(self, plan: dict, *, kind: str = "check") -> str:
        """Caller owns the transaction when reserving an experiment attempt."""
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-") + uuid.uuid4().hex[:10]
        stamp = now()
        self.db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)", (
            run_id, plan["project_id"], plan["recipe_id"], kind,
            "planned", stamp, stamp, json.dumps(plan, sort_keys=True), None,
        ))
        self.db.execute("INSERT INTO events VALUES (NULL,?,?,?,?)", (
            run_id, stamp, "planned", "Immutable plan captured",
        ))
        return run_id

    def transition(self, run_id: str, status: str, result: dict | None = None):
        allowed = {"planned": {"running", "blocked", "recorded"},
                   "running": {"passed", "failed", "interrupted"}}
        with self.db:
            current = self.db.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()
            if current is None or status not in allowed.get(current["status"], set()):
                raise ValueError(f"Invalid run transition to {status}")
            updated = self.db.execute(
                "UPDATE runs SET status=?,updated_at=?,result=? WHERE id=? AND status=?",
                (status, now(), json.dumps(result) if result is not None else None,
                 run_id, current["status"]),
            )
            if updated.rowcount != 1:
                raise ValueError("Run changed concurrently")
            self.db.execute("INSERT INTO events VALUES (NULL,?,?,?,?)", (
                run_id, now(), status, json.dumps(result or {}),
            ))

    def run(self, run_id: str) -> dict:
        row = self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown run: {run_id}")
        result = dict(row)
        result["plan"] = json.loads(result["plan"])
        result["result"] = json.loads(result["result"]) if result["result"] else None
        result["artifacts"] = [dict(r) for r in self.db.execute(
            "SELECT * FROM artifacts WHERE run_id=? ORDER BY id", (run_id,))]
        result["events"] = [dict(r) for r in self.db.execute(
            "SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,))]
        return result

    def runs(self) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT id,project_id,recipe_id,kind,status,created_at FROM runs ORDER BY created_at DESC")]

    def attach(self, run_id: str, artifact: dict):
        self.run(run_id)
        with self.db:
            self.db.execute("INSERT INTO artifacts VALUES (NULL,?,?,?,?,?,?,?)", (
                run_id, artifact["path"], artifact["sha256"], artifact["bytes"],
                artifact["role"], artifact["evidence_status"], now(),
            ))

    def update_project(self, project_id: str, fields: dict):
        current = self.overrides().get(project_id, {}) | fields
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO project_overrides VALUES (?,?,?)", (
                project_id, json.dumps(current), now(),
            ))

    def overrides(self) -> dict:
        return {r["project_id"]: json.loads(r["fields"]) for r in self.db.execute(
            "SELECT * FROM project_overrides")}

    def decide(self, project_id: str, text: str, run_id: str | None):
        if run_id and self.run(run_id)["project_id"] != project_id:
            raise ValueError("Decision run belongs to a different project")
        if not text.strip():
            raise ValueError("Decision text cannot be empty")
        with self.db:
            self.db.execute("INSERT INTO decisions VALUES (NULL,?,?,?,?)", (
                project_id, run_id, text, now(),
            ))

    def decisions(self, project_id: str) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM decisions WHERE project_id=? ORDER BY id", (project_id,))]
