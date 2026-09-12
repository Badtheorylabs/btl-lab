"""Build clean source snapshots and exercise only the installed wheel packages."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def verify(siblings: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME", "BTL_WORKSPACE"}}
    env.update(PIP_DISABLE_PIP_VERSION_CHECK="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    commands = []

    def run(argv, *, cwd=output, expected=0, structured=False):
        result = subprocess.run([str(x) for x in argv], cwd=cwd, env=env, capture_output=True,
                                text=True, timeout=180)
        commands.append({"command": [str(x) for x in argv], "returncode": result.returncode})
        (output / f"command-{len(commands):02d}.log").write_text(result.stdout + result.stderr)
        if result.returncode != expected:
            raise RuntimeError(f"{argv}: expected exit {expected}, got {result.returncode}: {result.stderr[-2000:]}")
        return json.loads(result.stdout) if structured else result.stdout

    wheels = output / "wheels"
    wheels.mkdir()
    for name in ("btl-train", "btl-rl", "btl-kernels", "btl-measure", "btl-lab"):
        source = siblings / name
        snapshot = output / "sources" / name
        names = subprocess.check_output(["git", "-C", str(source), "ls-files", "-co", "--exclude-standard", "-z"])
        for relative in sorted(set(names.decode().split("\0")) - {""}):
            path = source / relative
            if not path.is_file() or path.is_symlink():
                continue
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--wheel-dir", wheels, snapshot])
    run([sys.executable, "-m", "venv", output / "env"])
    python = output / "env/bin/python"
    run([python, "-m", "pip", "install", "--no-index", "--no-deps", *sorted(wheels.glob("*.whl"))])
    run([python, "-c", "import importlib.util; assert importlib.util.find_spec('numpy') is None; "
                         "assert importlib.util.find_spec('torch') is None"])
    # Renaming the build inputs catches imports that still reach into source checkouts.
    (output / "sources").rename(output / "unused-sources")
    cli = output / "env/bin/btl"
    workspace = output / "workspace with spaces"
    run([cli, "--help"])
    run([output / "env/bin/btl-rl", "--help"])
    run([output / "env/bin/btl-measure", "--help"])
    run([cli, "init", workspace])
    prefix = [cli, "--workspace", workspace, "--json"]
    run(prefix + ["doctor"])
    run(prefix + ["doctor", "--engine", "advance"], expected=2)
    audit = run(prefix + ["run", "registry-audit"], structured=True)
    run(prefix + ["runs", audit["id"], "--verify"])
    measurement = run(prefix + ["measure", "compare", "examples/baseline.json", "examples/candidate.json"], structured=True)
    run(prefix + ["runs", measurement["id"], "--verify"])
    study = run(prefix + ["research", "create", "examples/study.json"], structured=True)["id"]
    run(prefix + ["research", "freeze", study])
    run(prefix + ["research", "ready", study], expected=2)
    for arm in ("baseline", "candidate"):
        run(prefix + ["research", "run", study, "--arm", arm])
    run(prefix + ["research", "ready", study])
    run(prefix + ["research", "compare", study])
    run(prefix + ["research", "decide", study, "--outcome", "inconclusive", "--reason", "Installation check only; audit timing is not a model result"])
    run(prefix + ["research", "report", study])
    run([cli, "init", workspace])
    report = {"passed": True, "commands": commands, "output": str(output), "workspace": str(workspace),
              "python": str(python), "numpy_installed": False, "gpu_frameworks_installed": False,
              "external_spend_usd": 0, "scope": "Installed-package functionality and isolation; no model execution"}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--siblings", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="btl-install-check-")) / "verification"
    report = verify(args.siblings.resolve(), output.absolute())
    print(json.dumps({key: value for key, value in report.items() if key != "commands"}, indent=2))


if __name__ == "__main__":
    main()
