# Research and model operations

The research engine owns questions, protocols, attempts, comparisons and decisions. Train owns explicit model-operation contracts. Both use Lab's existing private SQLite ledger and immutable run plans. The control packages remain small and use no GPU libraries.

For a new installation, run `btl init PATH` and work from that directory. The installed `btl` command supports the same workflows as the original workspace launcher. `btl doctor` diagnoses setup before a run. See the [installation guide](README.md).

## Start with a research question

A draft needs only a project, title and question. Save a workspace-local JSON document in this shape, using a project ID from `btl projects`:

```json
{
  "schema_version": 1,
  "project_id": "workspace/btl-train",
  "title": "Compare agentic RL foundations",
  "question": "Which compatible engine reaches the fixed quality target at lower total cost?",
  "literature": []
}
```

From the workspace root, run `./btl research create` followed by that file's relative path. The returned experiment ID is used by the following commands:

```text
btl research list
btl research show EXPERIMENT_ID
btl research amend EXPERIMENT_ID UPDATED_DRAFT_FILE
btl research freeze EXPERIMENT_ID
btl research ready EXPERIMENT_ID
btl research link-run EXPERIMENT_ID RUN_ID --role evaluation
btl research link-run EXPERIMENT_ID RUN_ID --role arm-observation --arm baseline
btl research run EXPERIMENT_ID --arm baseline
btl research run EXPERIMENT_ID --arm candidate
btl research compare EXPERIMENT_ID
btl research decide EXPERIMENT_ID --outcome inconclusive --reason "Explain the evidence and remaining uncertainty"
btl research report EXPERIMENT_ID
```

Use `./btl` when invoking the workspace launcher. Uppercase arguments above stand for your actual IDs and file paths. Draft amendment saves the old and new versions in event history. Literature is supplied reference metadata; the engine does not automatically fetch papers or verify their claims.

Freezing requires a hypothesis, a falsification condition, baseline and candidate recipe IDs, declared matched controls, the primary scalar metric and its direction, minimum relative improvement, repeats per arm, an evidence class, required evidence roles, and a zero external-spend cap. The complete protocol fields are:

```text
schema_version, project_id, title, question, hypothesis, falsification,
arms.baseline.recipe, arms.candidate.recipe, controls,
metric.field, metric.direction [min|max], metric.minimum_relative_improvement,
repeats [1..100], evidence_class [integrity|systems|behavioral],
required_evidence_roles [names, default evaluation],
external_spend_cap_usd [0], inputs [workspace file paths], literature [references]
```

The freeze captures recipe definitions, control-code hashes and declared input-file hashes. They cannot be changed silently afterward. Changing the experimental design after freezing requires a new experiment. Only the existing local integrity-check recipes can execute through `research run` in this milestone. A GPU recipe is rejected before an attempt launches.

Attempts are reserved transactionally. Failed attempts consume the per-arm limit, and the runner cannot keep sampling until it finds a favorable result. Imported historical receipts are separate ledger records and cannot be substituted for an experiment's newly executed observations. OS-killed attempts remain in progress until explicitly reconciled; automatic crash recovery is not implemented.

`research ready` audits the frozen protocol, implementation hashes, inputs, arm observations, attached artifact hashes, supporting evidence roles and linked operation state without executing anything. `research link-run` attaches a run already created by Adapt, Advance, Measure or another registered executor. An arm link consumes one replicate; a supporting link such as `evaluation` does not. Linking never alters the source run's result. Comparison requires all planned repeats to pass, intact result artifacts, the declared evidence class and numeric metric, unchanged inputs/recipes, and matching recorded host platform. It reports raw observations, means and relative improvement. These are descriptive statistics, not significance tests. Declared controls are part of the protocol; the engine does not independently certify every hardware or methodological control. Acceptance or rejection requires readiness plus the frozen criterion. Missing or invalid evidence supports an inconclusive decision, not a successful claim. Decisions are final for that experiment.

Reports are versioned JSON files in private Lab state. They include the protocol, event history, runs, linked operations and evidence references, readiness, comparison and decision. A report is an evidence package, not an automatically publishable paper.

## Choose a specific model operation

`./btl operation types` lists pretraining, continued pretraining, SFT, preference optimization, distillation, RL, inference, evaluation and export. Each has required input roles and candidate backends. Catalog membership does not imply model support.

An operation specification declares:

- `schema_version: 1`, `project_id`, `name`, `operation` and `backend`.
- `model` and `engine`: each has an `id` and a full 40-character commit or 64-character SHA-256 `revision`. Mutable names such as `main` are rejected. These pins are recorded, not remotely resolved by preflight.
- `inputs`: named roles with a workspace-local `path` and `sha256`. Use immutable manifests when payloads are large. SFT, pretraining and distillation require data and evaluation roles; inference requires requests; export requires a checkpoint.
- `context_length`, `effective_batch_size` and `precision` for non-export operations.
- `budget.wall_seconds` and `budget.external_spend_usd`. These describe the intended future execution budget and are not spending authorization or an implemented provider cap.
- Distillation additionally needs a pinned teacher and a `teacher-text` or `on-policy-kl` objective. RL needs an explicit objective and versioned taskset/harness/reward contract. Preference optimization needs its objective; export needs its target format.

Use `./btl operation plan SPEC_FILE --experiment EXPERIMENT_ID` to link a model plan to a research question. Omit `--experiment` for a standalone operation. The project IDs must match. Then use `./btl operation preflight PLAN_RUN_ID` to create a separate input-check run linked to that plan. Linking an operation does not make it an experimental arm or include it in quantitative comparisons.

Input preflight hashes local files under a default total 64 MiB read budget, detects drift and path escapes, and writes a verifiable receipt. `--max-hash-mb` explicitly changes that bound. It does not verify dataset semantics, remote model existence, loading, gradients, CUDA support, checkpoint recovery or quality. Its receipt always says `training_ready: false`.

Model execution, output evaluation, artifact export and paid-resource control remain unimplemented stages. The current stage support is returned with every plan. An input-preflight pass must never be described as a training or capability result.

## Current validation and research intake

The control suite covers registry/planner behavior, lifecycle, budget, drift, evidence-class, operation contracts, evidence links and Adapt/Advance/Measure controls. The installed-package verifier also exercises workspace initialization and the complete local research lifecycle without a source checkout. The original private BTL ledger contains four draft research questions; new workspaces start with no recorded experiments. No scientific comparison or capability gain is established by these control checks.

## Specialized supervised execution

The `btl adapt` entry point implements BTL Adapt's Unsloth-backed SFT worker with its own strict recipe. Generic operation plans above remain planning/input checks. See [the fine-tuning engine](https://github.com/Badtheorylabs/btl-train/blob/main/FINETUNE.md) for the separate execution contract and hardware-validation boundary. A Qwen3.5-0.8B A100 profile has passed bounded execution and recovery checks; production support is still gated.
