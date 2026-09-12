from __future__ import annotations

import hashlib
import json
import math
import statistics
import uuid
from pathlib import Path

import btl_kernels.manifest as kernel_manifest
import btl_rl.advance_local as advance_local
import btl_rl.prime_rl as prime_rl
import btl_rl.process as rl_process
import btl_train.operations as train_operations
import btl_rl.artifacts as rl_artifacts
import btl_rl.evaluation as rl_evaluation
import btl_rl.runtime as rl_runtime
import btl_measure.core as measure_core

from .checks import artifact_record, verify_artifacts
from .planner import plan
from .runner import execute_check
from .store import Store, now
from .workspace import Workspace, inside, sha256


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def implementation_sources() -> dict:
    package = Path(__file__).parent
    sources = {"btl_lab/" + name: package / name for name in
               ("research.py", "runner.py", "checks.py", "planner.py", "store.py", "workspace.py",
                "advance.py", "measure.py", "doctor.py", "finetune.py", "runtime_probe.py", "workflow_cli.py")}
    sources["btl_kernels/manifest.py"] = Path(kernel_manifest.__file__)
    sources["btl_rl/advance_local.py"] = Path(advance_local.__file__)
    sources["btl_rl/prime_rl.py"] = Path(prime_rl.__file__)
    sources["btl_rl/process.py"] = Path(rl_process.__file__)
    sources["btl_train/operations.py"] = Path(train_operations.__file__)
    sources["btl_rl/artifacts.py"] = Path(rl_artifacts.__file__)
    sources["btl_rl/evaluation.py"] = Path(rl_evaluation.__file__)
    sources["btl_rl/runtime.py"] = Path(rl_runtime.__file__)
    sources["btl_measure/core.py"] = Path(measure_core.__file__)
    return {name: sha256(path) for name, path in sources.items()}


class Research:
    def __init__(self, workspace: Workspace, store: Store):
        self.workspace, self.store, self.db = workspace, store, store.db
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL, title TEXT NOT NULL,
                status TEXT NOT NULL, spec TEXT NOT NULL, frozen TEXT, protocol_hash TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, decision TEXT
            );
            CREATE TABLE IF NOT EXISTS experiment_runs (
                experiment_id TEXT NOT NULL REFERENCES experiments(id),
                arm TEXT NOT NULL, replicate INTEGER NOT NULL,
                run_id TEXT NOT NULL UNIQUE REFERENCES runs(id),
                PRIMARY KEY(experiment_id,arm,replicate)
            );
            CREATE TABLE IF NOT EXISTS experiment_events (
                id INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id),
                created_at TEXT NOT NULL, event TEXT NOT NULL, detail TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiment_links (
                id INTEGER PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id),
                run_id TEXT NOT NULL REFERENCES runs(id), role TEXT NOT NULL,
                arm TEXT, note TEXT NOT NULL, created_at TEXT NOT NULL,
                UNIQUE(experiment_id, run_id, role, arm)
            );
        """)

    def validate_draft(self, spec: dict):
        if not isinstance(spec, dict) or spec.get("schema_version") != 1:
            raise ValueError("Research draft requires schema_version=1")
        self.workspace.project(spec["project_id"])
        for key in ("title", "question"):
            if not isinstance(spec.get(key), str) or not spec[key].strip():
                raise ValueError(f"Research draft requires {key}")
        fingerprint(spec)
        required = {"hypothesis", "falsification", "arms", "controls", "metric", "repeats",
                    "evidence_class", "external_spend_cap_usd"}
        if required.issubset(spec):
            self.validate(spec)

    def validate(self, spec: dict):
        if spec.get("schema_version") != 1:
            raise ValueError("Research protocol requires schema_version=1")
        self.workspace.project(spec["project_id"])
        for key in ("title", "question", "hypothesis", "falsification"):
            if not isinstance(spec.get(key), str) or not spec[key].strip():
                raise ValueError(f"Research protocol requires {key}")
        arms = spec.get("arms", {})
        if set(arms) != {"baseline", "candidate"}:
            raise ValueError("Declare baseline and candidate arms")
        for arm in arms.values():
            self.workspace.recipe(arm["recipe"])
        if not isinstance(spec.get("controls"), dict) or not spec["controls"]:
            raise ValueError("Declare the matched controls")
        if spec.get("evidence_class") not in {"integrity", "systems", "behavioral"}:
            raise ValueError("Declare integrity, systems or behavioral evidence_class")
        metric = spec["metric"]
        if metric.get("direction") not in {"min", "max"} or not metric.get("field"):
            raise ValueError("Metric needs a field and min/max direction")
        threshold = metric.get("minimum_relative_improvement", 0)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("Improvement threshold must be finite and nonnegative")
        if type(spec.get("repeats")) is not int or not 1 <= spec["repeats"] <= 100:
            raise ValueError("repeats must be an integer between 1 and 100")
        if type(spec.get("external_spend_cap_usd")) not in {int, float} or spec["external_spend_cap_usd"] != 0:
            raise ValueError("This local research executor supports zero external spend only")
        required_roles = spec.get("required_evidence_roles", ["evaluation"])
        if not isinstance(required_roles, list) or not required_roles or any(
            not isinstance(role, str) or not role.strip() for role in required_roles
        ):
            raise ValueError("required_evidence_roles must be a nonempty list of names")
        fingerprint(spec)

    def event(self, experiment_id: str, event: str, detail: dict):
        self.db.execute("INSERT INTO experiment_events VALUES (NULL,?,?,?,?)", (
            experiment_id, now(), event, json.dumps(detail),
        ))

    def create(self, spec: dict) -> dict:
        self.validate_draft(spec)
        experiment_id = "exp-" + uuid.uuid4().hex[:12]
        with self.db:
            self.db.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?)", (
                experiment_id, spec["project_id"], spec["title"], "draft",
                json.dumps(spec), None, None, now(), now(), None,
            ))
            self.event(experiment_id, "created", {"scope": "Protocol only; no execution", "spec": spec})
        return self.get(experiment_id)

    def amend(self, experiment_id: str, spec: dict) -> dict:
        self.validate_draft(spec)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            current = self.get(experiment_id)
            if current["status"] != "draft":
                raise ValueError("Only drafts can change; create a new experiment after freezing")
            if spec["project_id"] != current["project_id"]:
                raise ValueError("An experiment cannot change project")
            self.db.execute("UPDATE experiments SET spec=?,title=?,updated_at=? WHERE id=?",
                            (json.dumps(spec), spec["title"], now(), experiment_id))
            self.event(experiment_id, "draft-amended", {"previous": current["spec"], "new": spec})
        return self.get(experiment_id)

    def get(self, experiment_id: str) -> dict:
        row = self.db.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown experiment: {experiment_id}")
        result = dict(row)
        for key in ("spec", "frozen", "decision"):
            result[key] = json.loads(result[key]) if result[key] else None
        result["runs"] = [dict(r) for r in self.db.execute(
            "SELECT * FROM experiment_runs WHERE experiment_id=? ORDER BY arm,replicate", (experiment_id,))]
        result["events"] = [dict(r) for r in self.db.execute(
            "SELECT * FROM experiment_events WHERE experiment_id=? ORDER BY id", (experiment_id,))]
        result["operations"] = [dict(r) for r in self.db.execute(
            "SELECT id,kind,status,plan FROM runs WHERE kind LIKE 'model-operation%'"
        ) if json.loads(r["plan"]).get("experiment_id") == experiment_id]
        for operation in result["operations"]:
            operation["plan"] = json.loads(operation["plan"])
        result["links"] = [dict(r) for r in self.db.execute(
            "SELECT * FROM experiment_links WHERE experiment_id=? ORDER BY id", (experiment_id,)
        )]
        return result

    def list(self) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT id,title,project_id,status,protocol_hash FROM experiments ORDER BY created_at DESC")]

    def freeze(self, experiment_id: str) -> dict:
        experiment = self.get(experiment_id)
        if experiment["status"] != "draft":
            raise ValueError("Only a draft can be frozen; create a new protocol for changes")
        spec = experiment["spec"]
        self.validate(spec)
        references = [artifact_record(self.workspace, path, "protocol-input", "reference-only")
                      for path in spec.get("inputs", [])]
        frozen = {"spec": spec, "inputs": references,
                  "implementation_sources": implementation_sources(),
                  "recipes": {arm: self.workspace.recipe(value["recipe"])
                              for arm, value in spec["arms"].items()},
                  "literature": spec.get("literature", [])}
        with self.db:
            changed = self.db.execute(
                "UPDATE experiments SET frozen=?,protocol_hash=?,status='frozen',updated_at=? WHERE id=? AND status='draft'",
                (json.dumps(frozen), fingerprint(frozen), now(), experiment_id),
            )
            if changed.rowcount != 1:
                raise ValueError("Protocol changed concurrently")
            self.event(experiment_id, "frozen", {"protocol_hash": fingerprint(frozen)})
        return self.get(experiment_id)

    def check_frozen(self, experiment: dict):
        frozen = experiment["frozen"]
        if not frozen or fingerprint(frozen) != experiment["protocol_hash"]:
            raise ValueError("Missing or corrupt frozen protocol")
        if frozen.get("implementation_sources") != implementation_sources():
            raise ValueError("Research implementation changed after freeze")
        for artifact in frozen["inputs"]:
            if sha256(inside(self.workspace.root, artifact["path"])) != artifact["sha256"]:
                raise ValueError(f"Frozen input changed: {artifact['path']}")
        for arm, recipe in frozen["recipes"].items():
            if self.workspace.recipe(recipe["id"]) != recipe:
                raise ValueError(f"Recipe changed after freeze: {arm}")

    def run(self, experiment_id: str, arm: str) -> dict:
        experiment = self.get(experiment_id)
        if experiment["status"] not in {"frozen", "running"}:
            raise ValueError("Freeze the protocol before running; decided experiments are closed")
        if arm not in experiment["spec"]["arms"]:
            raise ValueError("Unknown arm")
        self.check_frozen(experiment)
        resolved = plan(self.workspace, experiment["spec"]["arms"][arm]["recipe"])
        if resolved["backend"] != "local-audit":
            raise ValueError("Only local audit recipes are executable here; model operations have a separate preflight")
        resolved |= {"implementation_project_id": resolved["project_id"],
                     "project_id": experiment["project_id"], "experiment_id": experiment_id,
                     "protocol_hash": experiment["protocol_hash"], "arm": arm}
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            current = self.get(experiment_id)
            if current["status"] not in {"frozen", "running"}:
                raise ValueError("Experiment no longer open")
            replicate = sum(r["arm"] == arm for r in current["runs"]) + 1
            if replicate > current["spec"]["repeats"]:
                raise ValueError("Frozen attempt budget exhausted; failed attempts count")
            resolved["replicate"] = replicate
            run_id = self.store.create_in_transaction(resolved, kind="research-check")
            self.db.execute("INSERT INTO experiment_runs VALUES (?,?,?,?)", (experiment_id, arm, replicate, run_id))
            self.db.execute("INSERT INTO experiment_links VALUES (NULL,?,?,?,?,?,?)", (
                experiment_id, run_id, "arm-observation", arm,
                f"Local check observation for {arm} replicate {replicate}", now(),
            ))
            self.db.execute("UPDATE experiments SET status='running',updated_at=? WHERE id=?", (now(), experiment_id))
            self.event(experiment_id, "attempt-reserved", {"run_id": run_id, "arm": arm, "replicate": replicate})
        return execute_check(self.workspace, self.store, run_id)

    def link_run(self, experiment_id: str, run_id: str, role: str, arm: str | None = None,
                 note: str = "") -> dict:
        """Attach an already recorded engine/evaluation run to a frozen study.

        Linking does not change the run's result. An arm link is an observation
        and consumes that arm's replicate budget; other roles, such as
        ``evaluation`` or ``literature-audit``, remain supporting evidence.
        """
        experiment = self.get(experiment_id)
        if experiment["status"] not in {"frozen", "running"}:
            raise ValueError("Freeze the experiment before linking evidence")
        run = self.store.run(run_id)
        if not isinstance(role, str) or not role.strip():
            raise ValueError("Evidence role cannot be empty")
        if arm is not None and arm not in experiment["spec"]["arms"]:
            raise ValueError(f"Unknown experiment arm: {arm}")
        if arm is not None:
            if role not in {"arm-observation", "model-operation", "baseline", "candidate"}:
                raise ValueError("An arm link needs role arm-observation, model-operation, baseline or candidate")
            used = [r for r in experiment["runs"] if r["arm"] == arm]
            if len(used) >= experiment["spec"]["repeats"]:
                raise ValueError(f"Attempt budget exhausted for arm {arm}")
            if any(r["run_id"] == run_id for r in experiment["runs"]):
                raise ValueError("Run is already linked as an arm observation")
            replicate = len(used) + 1
        else:
            replicate = None
        note = note.strip() or f"Linked {role} run {run_id}"
        with self.db:
            duplicate = self.db.execute(
                "SELECT 1 FROM experiment_links WHERE experiment_id=? AND run_id=? AND role=? AND arm IS ?",
                (experiment_id, run_id, role.strip(), arm),
            ).fetchone()
            if duplicate is not None:
                raise ValueError("This run/evidence role is already linked")
            try:
                self.db.execute(
                    "INSERT INTO experiment_links VALUES (NULL,?,?,?,?,?,?)",
                    (experiment_id, run_id, role.strip(), arm, note, now()),
                )
            except Exception as error:
                if "UNIQUE" in str(error).upper():
                    raise ValueError("This run/evidence role is already linked") from error
                raise
            if arm is not None:
                self.db.execute(
                    "INSERT INTO experiment_runs VALUES (?,?,?,?)",
                    (experiment_id, arm, replicate, run_id),
                )
            self.event(experiment_id, "evidence-linked", {
                "run_id": run_id, "role": role, "arm": arm,
                "run_status": run["status"], "run_kind": run["kind"],
            })
        return self.get(experiment_id)

    def readiness(self, experiment_id: str) -> dict:
        """Return blockers for a decision without changing experiment state."""
        experiment = self.get(experiment_id)
        blockers: list[str] = []
        if experiment["status"] == "draft":
            blockers.append("Protocol is still a draft")
        if experiment["status"] == "decided":
            blockers.append("Experiment is already decided")
        if experiment["frozen"]:
            try:
                self.check_frozen(experiment)
            except ValueError as error:
                blockers.append(str(error))
        else:
            try:
                self.validate(experiment["spec"])
            except ValueError as error:
                blockers.append(str(error))
        arms: dict[str, dict] = {}
        for arm in ("baseline", "candidate"):
            links = [r for r in experiment["runs"] if r["arm"] == arm]
            statuses = []
            valid = 0
            for link in links:
                run = self.store.run(link["run_id"])
                statuses.append({"run_id": run["id"], "status": run["status"], "kind": run["kind"]})
                if run["status"] == "passed" and verify_artifacts(self.workspace, run)["passed"]:
                    valid += 1
                else:
                    blockers.append(f"Arm {arm} has incomplete or failed run {run['id']}")
            required = experiment["spec"].get("repeats", 0)
            if len(links) < required:
                blockers.append(f"Arm {arm} has {len(links)}/{required} linked observations")
            arms[arm] = {"required": required, "linked": len(links), "valid": valid, "runs": statuses}
        required_roles = experiment["spec"].get("required_evidence_roles", ["evaluation"])
        evidence: dict[str, int] = {}
        evidence_details: dict[str, list[dict]] = {}
        for link in experiment["links"]:
            run = self.store.run(link["run_id"])
            integrity = verify_artifacts(self.workspace, run)
            valid = run["status"] == "passed" and integrity["passed"]
            evidence[link["role"]] = evidence.get(link["role"], 0) + 1
            evidence_details.setdefault(link["role"], []).append({
                "run_id": run["id"], "arm": link["arm"], "status": run["status"],
                "kind": run["kind"], "valid": valid,
                "artifact_integrity": integrity,
            })
            if not valid:
                blockers.append(f"Evidence link {link['role']} has incomplete or changed run {run['id']}")
        for role in required_roles:
            if not isinstance(role, str) or not role:
                blockers.append("Protocol contains an invalid required evidence role")
            elif evidence.get(role, 0) == 0:
                blockers.append(f"Missing required evidence role: {role}")
            elif not any(item["valid"] for item in evidence_details.get(role, [])):
                blockers.append(f"Required evidence role {role} has no complete passed run")
        running = [r for r in experiment["operations"] if r["status"] in {"planned", "running"}]
        if running:
            blockers.append("A linked model operation is still planned or running")
        return {"experiment_id": experiment_id, "ready_for_decision": not blockers,
                "blockers": list(dict.fromkeys(blockers)), "arms": arms, "evidence": evidence,
                "evidence_details": evidence_details,
                "required_evidence_roles": required_roles,
                "scope": "Readiness audit under the frozen protocol; no model execution or promotion"}

    def compare(self, experiment_id: str) -> dict:
        experiment = self.get(experiment_id)
        if not experiment["frozen"]:
            raise ValueError("Freeze the protocol before comparison")
        spec = experiment["spec"]
        reasons, values, hosts = [], {"baseline": [], "candidate": []}, set()
        try:
            self.check_frozen(experiment)
        except ValueError as error:
            reasons.append(str(error))
        for link in experiment["runs"]:
            run = self.store.run(link["run_id"])
            result = run["result"] or {}
            if run["status"] != "passed" or not verify_artifacts(self.workspace, run)["passed"]:
                reasons.append(f"Failed, incomplete or changed run: {run['id']}")
                continue
            if result.get("evidence_class") != spec["evidence_class"]:
                reasons.append(f"Wrong evidence class: {run['id']}")
                continue
            value = result.get(spec["metric"]["field"])
            if type(value) not in {float, int} or not math.isfinite(value):
                reasons.append(f"Missing/non-finite metric: {run['id']}")
                continue
            hosts.add(fingerprint(run["plan"]["host"]))
            values[link["arm"]].append(value)
        if len(hosts) > 1:
            reasons.append("Host platform differs between runs")
        for arm in values:
            if len(values[arm]) != spec["repeats"]:
                reasons.append(f"Incomplete valid replicates for {arm}")
        report = {"experiment_id": experiment_id, "protocol_hash": experiment["protocol_hash"],
                  "comparable": not reasons, "reasons": reasons, "values": values,
                  "evidence_class": spec["evidence_class"],
                  "readiness": self.readiness(experiment_id),
                  "scope": "Descriptive comparison under the declared protocol; not automatic promotion or statistical significance"}
        if not reasons:
            means = {arm: statistics.mean(numbers) for arm, numbers in values.items()}
            baseline = means["baseline"]
            delta = means["candidate"] - baseline
            if spec["metric"]["direction"] == "min":
                delta = -delta
            improvement = delta / abs(baseline) if baseline != 0 else None
            report |= {"means": means, "relative_improvement": improvement,
                       "meets_declared_threshold": improvement is not None and improvement > 0
                       and improvement >= spec["metric"].get("minimum_relative_improvement", 0)}
        return report

    def decide(self, experiment_id: str, outcome: str, rationale: str) -> dict:
        if outcome not in {"accept", "reject", "inconclusive"} or not rationale.strip():
            raise ValueError("A decision requires an outcome and rationale")
        experiment = self.get(experiment_id)
        if experiment["status"] not in {"frozen", "running"}:
            raise ValueError("Experiment is not open for a decision")
        report = self.compare(experiment_id)
        readiness = report["readiness"]
        if outcome in {"accept", "reject"} and not readiness["ready_for_decision"]:
            raise ValueError("Decision requires complete linked evidence: " + "; ".join(readiness["blockers"]))
        if outcome == "accept" and (not report["comparable"] or not report.get("meets_declared_threshold")):
            raise ValueError("Acceptance requires complete evidence meeting the frozen threshold")
        if outcome == "reject" and not report["comparable"]:
            raise ValueError("Incomplete evidence supports inconclusive, not rejection")
        decision = {"outcome": outcome, "rationale": rationale, "comparison": report}
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if self.get(experiment_id)["status"] not in {"frozen", "running"}:
                raise ValueError("Experiment already decided")
            if any(self.store.run(r["run_id"])["status"] in {"planned", "running"}
                   for r in self.get(experiment_id)["runs"]):
                raise ValueError("An attempt is still in progress")
            self.db.execute("UPDATE experiments SET status='decided',decision=?,updated_at=? WHERE id=?",
                            (json.dumps(decision), now(), experiment_id))
            self.event(experiment_id, "decided", decision)
        return self.get(experiment_id)
