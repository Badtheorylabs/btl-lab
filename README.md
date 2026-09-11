# BTL Lab

BTL Lab provides one local entry point for projects, recipes, run records, decisions and artifact integrity. The first implementation connects the existing workspace without moving research or importing GPU frameworks into the control process. BTL is the lab; Tinfield is the model family. The environment/data product name remains unselected.

The [research and model-operation workflows](WORKFLOWS.md) now support question drafts, amendments, frozen protocols, bounded local attempts, comparisons and decisions, plus explicit model-operation plans, input-integrity preflight, cross-lane run links and readiness audits. The specialized BTL Adapt profile has a qualified A100 execution path; other GPU operations remain unimplemented.

## Start

From the BTL workspace:

```sh
./btl status
./btl layout
./btl projects
./btl recipes
./btl backends
./btl research list
./btl operation types
./btl adapt --help
./btl advance local --help
./btl measure --help
./btl run registry-audit
./btl run kernel-source-audit
./btl runs
```

Python 3.11 or newer is required. The workspace launcher tries an installed Python 3.13/3.12/3.11 if `python3` is older. It downloads nothing. The implementation uses the Python standard library plus the sibling BTL Train and BTL Kernels packages. It adds no model weights, GPU packages or virtual environment. The package also declares a `btl` console entry point for a later installation.

For a standalone editable checkout, clone the four stack repositories and install the local packages together:

```sh
git clone https://github.com/Badtheorylabs/btl-train.git
git clone https://github.com/Badtheorylabs/btl-kernels.git
git clone https://github.com/Badtheorylabs/btl-measure.git
git clone https://github.com/Badtheorylabs/btl-lab.git
python3.13 -m pip install -e btl-train -e btl-kernels -e btl-measure -e btl-lab
```

Use `./btl --json projects` for structured output. `--json` and `--workspace` precede the subcommand. The CLI discovers the workspace from the current directory or its parents; `BTL_WORKSPACE` and `--workspace` support explicit selection.

## Research workflow

```sh
./btl projects --program learning
./btl project research/tinfield_learning_2026-09-08
./btl project research/tinfield_learning_2026-09-08 --next-gate "Review the existing estimator result before another experiment"
./btl evidence --project research/tinfield_learning_2026-09-08 --file research/tinfield_learning_2026-09-08/RESULT_LOCAL_CREDIT_08B.md --label existing-local-credit-result
./btl decision workspace/btl-train "Prime-RL is the preferred RL foundation; exact-workload validation remains required"
```

Project updates are overlays in SQLite, preserving the original discovery record. Owners stay unassigned until explicitly set with `--owner`. A project lifecycle and the quality of its evidence are separate fields. An imported receipt receives `recorded` / `reported-unreviewed`, never a new training success. Attaching it hashes the file and records its location without copying large artifacts.

`./btl runs RUN_ID` shows the saved plan, result, events and artifact references. `./btl runs RUN_ID --verify` checks whether the attached files still match their recorded hashes. Replace `RUN_ID` with an ID from `./btl runs`. Hash integrity is not independent reproduction of a receipt's claims. A decision can cite a run with `--run RUN_ID`; a run from another project is rejected.

## Prime-RL integration

Prime-RL is the preferred RL foundation. The initial adapter creates a plan for the upstream reverse-text recipe at commit `dad79d1ce85390a2c818416ef100677f53f8c1ec`. This is an integration baseline using an upstream SFT-warmed Qwen3-0.6B model, not a Tinfield capability benchmark.

```sh
./btl plan prime-rl-reverse-text --save
```

Pass `--checkout` with the path to an existing pinned Prime-RL checkout for source checks. The planner records the actual commit, tracked-change status, config hash and parsed TOML, submodule pins, host information and a native `--dry-run` command. It does not clone, install, download weights, run the native command or launch training. Even a matching checkout remains blocked for GPU execution until the execution adapter and exact hardware recipe are validated.

The other lanes are represented in `backends.json` with explicit implementation status. **BTL Adapt** is the fine-tuning engine; Unsloth Core is its current CUDA backend. The Qwen3.5-0.8B A100 profile has passed bounded execution and recovery checks, but the production recipe remains unqualified. Checkpoint handoff to other engines, inference serving and automatic publication are separate stages.

## Local checks and storage

- `registry-audit` verifies registered project/document paths.
- `kernel-source-audit` verifies the existing Qwen kernel review manifest's source hashes. It executes no kernels.
- Plans and check runs have independent IDs and immutable serialized plan snapshots.
- Check events progress from planned to running to passed/failed/interrupted. A blocked backend records its blockers and exits nonzero.
- `private/workspace/lab-state/lab.sqlite3` stores runs, decisions, project overlays and artifact references. Its `results/` directory stores local audit receipts. These paths are outside this repository; `.btl-workspace.json` locates them. Preserve this private state when retaining run history.

`private/workspace/projects.json` now holds the original 42 discovery entries plus the shared Lab, Train and Kernels projects. The old `projects.json` and `.state` paths are ignored compatibility links. Existing entries remain untriaged unless reviewed; directory names and document existence are not capability evidence. Program assignments from the original scan remain tentative.

## Validation

```sh
cd platform/btl-lab
python3.13 -m pytest -q
```

Eighty-two tests cover run transitions, artifact drift, source corruption, missing paths, path escapes, immutable plan snapshots, evidence import, project overlays, revision mismatch, dirty checkouts, grouped/private workspace paths, blocked GPU launch, research lifecycle, operation contracts, BTL Advance controls, BTL Measure comparisons, evidence links and readiness checks. The actual workspace path audit and source-manifest audit passed. BTL Adapt and BTL Advance qualification receipts are recorded separately; the Lab control implementation itself does not claim a scientific benchmark result.

Use [the foundation comparison](FOUNDATION_REVIEW_2026-09-09.md) and [experiment template](EXPERIMENT_TEMPLATE.md) for the broader program. The next implementation gate is a pinned Prime-RL execution profile with resource authorization, bounded runtime, checkpoint recovery and held-out evaluation.

The project registry, state database, evidence files and disk-audit material are private workspace records. Publish from an explicit reviewed file list. `.gitignore` is a convenience, not a complete publication review.

## Repository boundary

This directory is the Lab code repository. The Prime-RL inspector lives in [BTL Train](https://github.com/Badtheorylabs/btl-train); the manifest checker lives in [BTL Kernels](https://github.com/Badtheorylabs/btl-kernels). The root workspace launcher adds these local packages to its import path. Private registries, SQLite state, source-review snapshots and disk audits stay in the local BTL workspace and are not release inputs.

The specialized `btl adapt` command provides BTL Adapt's Unsloth-backed supervised worker. Its Qwen3.5-0.8B A100 profile passed execution, adapter serialization, resume, fresh-process reload and fixed-task behavior-retention checks. It has no demonstrated capability gain or speedup claim. See [the fine-tuning guide](https://github.com/Badtheorylabs/btl-train/blob/main/FINETUNE.md).

The specialized `btl advance local` command provides BTL Advance's no-spend MLX contextual-bandit RL worker. Its Qwen3.5-0.8B probe passed policy sampling, verifier rewards, group-relative credit, finite gradients, adapter updates and split-process checkpoint recovery. It is mechanics evidence only; Prime-RL agentic execution remains the scale path.

The specialized `btl measure` command provides BTL Measure's independent per-item evaluation checks and paired comparisons. It recomputes scores, rejects taskset/model mismatches and records regressions without trusting aggregate claims or automatically promoting a release.

BTL-owned code in this repository is released under the MIT License. Referenced upstream projects and model artifacts retain their own licenses.
