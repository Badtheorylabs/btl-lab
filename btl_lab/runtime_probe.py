"""Adapt environment diagnosis in the selected worker interpreter."""
import json
import sys

from btl_train.finetune_worker import doctor


def main() -> int:
    config = json.load(sys.stdin)
    result = doctor(config)
    if result["ready_for_worker_start"]:
        try:
            import torch
            result["cuda_devices"] = torch.cuda.device_count()
            if not torch.cuda.is_available() or result["cuda_devices"] != 1:
                result["blockers"].append("Adapt requires exactly one visible CUDA device")
            elif config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
                result["blockers"].append("The selected CUDA device cannot execute the BF16 recipe")
        except (ImportError, RuntimeError, OSError) as error:
            result["blockers"].append(f"CUDA initialization failed: {error}")
    result["ready_for_worker_start"] = not result["blockers"]
    print(json.dumps(result, indent=2))
    return 0 if result["ready_for_worker_start"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
