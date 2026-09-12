# BTL Lab

BTL Lab connects research questions, model runs, evaluation records and decisions in one local workspace. BTL Adapt performs supervised fine-tuning, BTL Advance owns RL, and BTL Measure compares evaluation records. Tinfield is the model family.

Version 0.2 adds portable workspace creation, runtime diagnosis, installed-package checks and an automatic Advance-to-Measure handoff. The control installation needs Python 3.11 or newer and the small BTL packages. Training frameworks belong to the selected worker environment.

## Install and start

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install 'btl-lab @ git+https://github.com/Badtheorylabs/btl-lab.git@main'
btl init ./tinfield
cd tinfield
btl doctor
btl run registry-audit
btl measure compare examples/baseline.json examples/candidate.json
```

The comparison files are handwritten format examples. Their scores say nothing about model capability. Initialization writes the project registry and recipe catalog under `.btl/`, with the ledger under `.btl/state/`. Repeating initialization preserves an existing workspace. Existing nonempty directories are refused. The original grouped BTL workspace layout also remains supported.

The published Lab package pins its sibling dependencies to exact Git commits. The standalone `btl` entry point works outside the source checkout; a private BTL registry or root launcher is not required.

## Run the supported local model recipe

On Apple Silicon, install the tested MLX profile into your environment:

```sh
python -m pip install 'btl-lab[mlx] @ git+https://github.com/Badtheorylabs/btl-lab.git@main'
btl doctor --engine advance --model /absolute/path/to/Qwen3.5-0.8B
btl advance local --model /absolute/path/to/Qwen3.5-0.8B --steps 6
```

The model must already exist locally. To reuse a separate MLX environment, pass `--python /path/to/environment/bin/python` to both commands. The launcher probes that interpreter, checks local assets, hashes model files, runs a bounded worker, verifies the final checkpoint and records the outputs. A successful run creates a separate BTL Measure record automatically. The response includes both run IDs.

A split run can be continued by ID:

```sh
btl advance local --model /absolute/path/to/Qwen3.5-0.8B --steps 3 --total-steps 6
btl advance resume RUN_ID
btl runs RUN_ID --verify
btl measure from-run RUN_ID
```

Replace `RUN_ID` with the recorded training run. Resume restores the latest complete checkpoint and checks recorded artifacts, model identity and worker source. The worker deadline is set by `--max-seconds`; the supervisor allows up to five additional seconds to terminate. Runtime probing and model hashing happen before the worker deadline. GPU resources are never provisioned by these commands.

The local profile is a synthetic two-action contextual bandit. The installed 0.2 workflow was exercised on Qwen3.5-0.8B: six uninterrupted steps and a three-step run resumed to six produced the same adapter hash. Each run generated a verifiable Measure report. This establishes local execution and recovery, while agentic RL and broader capability remain unqualified.

## Supervised training and research

`btl doctor --engine adapt --python GPU_PYTHON --recipe RECIPE.json` checks pinned backend versions, CUDA availability and the single-device profile. Adapt requires accepted tokenized inputs and its explicit execution recipe. See [BTL Train](https://github.com/Badtheorylabs/btl-train/blob/main/FINETUNE.md). The earlier A100 execution qualification does not establish a speedup or production support.

For research, start with `btl research create examples/study.json`, then freeze, run its baseline and candidate arms, inspect readiness, compare and decide. That example exercises integrity bookkeeping only. See [the workflow guide](WORKFLOWS.md) for real protocols, input hashes, supporting evidence and claim boundaries.

Prime-RL inspection is available through `btl plan prime-rl-reverse-text --checkout PATH`. Its GPU execution adapter remains unimplemented. [BTL RL](https://github.com/Badtheorylabs/btl-rl) owns that integration; [BTL Measure](https://github.com/Badtheorylabs/btl-measure) owns record validation; [BTL Kernels](https://github.com/Badtheorylabs/btl-kernels) provides source integrity checks.

## Verify a change

Clone the five stack repositories side by side. For source development, install the local packages explicitly with `--no-deps`, and install pytest and NumPy for the control tests:

```sh
python -m pip install pytest numpy setuptools wheel
python -m pip install --no-deps -e ../btl-train -e ../btl-rl -e ../btl-kernels -e ../btl-measure -e .
python -m pytest -q
python tools/verify_install.py
```

The install verifier builds clean source snapshots, installs wheels in a fresh environment without NumPy or GPU frameworks, then exercises initialization, diagnosis, checks, evaluation, the complete research lifecycle and artifact verification. It retains logs and a JSON receipt at the printed output path. Separate model and hardware checks are required for changes to a training backend.

BTL-owned code is MIT licensed. Upstream projects, model weights, tasksets and artifacts retain their own licenses. Private registries, credentials and run outputs are not repository inputs.
