"""Pure-GPU single-node model-parallel loading for Response-G1 probe jobs."""
from __future__ import annotations
from collections import Counter
from typing import Any
import torch
from transformers import Qwen3VLForConditionalGeneration
_DEVICE_MAPS = ("auto", "balanced", "balanced_low_0", "sequential")

def add_model_parallel_args(parser: Any) -> None:
    parser.add_argument("--device_map", choices=_DEVICE_MAPS, default="balanced", help=("Transformers model-parallel placement. With two or more visible GPUs, 'balanced' distributes model weights across them; with one GPU the loader uses 'auto'."))
    parser.add_argument("--gpu_memory_fraction", type=float, default=0.85, help=("Fraction of currently free VRAM made available to model weights on each visible GPU when using two or more GPUs. The remaining VRAM is reserved for video tokens and activations; CPU/disk offload is never enabled."))

def _max_memory_for_visible_gpus(fraction: float) -> dict[int, str]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("--gpu_memory_fraction must be strictly between 0 and 1.")
    gib = 1024 ** 3
    budgets: dict[int, str] = {}
    for device_id in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(device_id)
        budget_gib = int((free_bytes / gib) * fraction)
        if budget_gib < 1:
            raise RuntimeError(f"GPU {device_id} has insufficient free memory ({free_bytes / gib:.2f} GiB).")
        budgets[device_id] = f"{budget_gib}GiB"
    return budgets

def _mapped_gpu_ids(model: Qwen3VLForConditionalGeneration) -> set[int]:
    mapped: set[int] = set()
    for device in model.hf_device_map.values():
        if isinstance(device, int):
            mapped.add(device)
        elif isinstance(device, str) and device.startswith("cuda:"):
            mapped.add(int(device.split(":", 1)[1]))
    return mapped

def load_qwen3vl(ckpt_path: str, args: Any) -> Qwen3VLForConditionalGeneration:
    """Load Qwen3-VL with balanced visible-GPU placement and no CPU/disk offload."""
    if not torch.cuda.is_available():
        raise RuntimeError("This probe implementation requires CUDA.")
    visible_gpus = torch.cuda.device_count()
    if visible_gpus == 1:
        device_map, max_memory = "auto", None
    else:
        device_map = args.device_map
        max_memory = _max_memory_for_visible_gpus(args.gpu_memory_fraction)
    kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "attn_implementation": "flash_attention_2", "device_map": device_map, "trust_remote_code": True}
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    model = Qwen3VLForConditionalGeneration.from_pretrained(ckpt_path, **kwargs).eval()
    device_map_result = model.hf_device_map
    forbidden = {str(device) for device in device_map_result.values()} & {"cpu", "disk"}
    if forbidden:
        raise RuntimeError(f"Model placement unexpectedly used {sorted(forbidden)}. This run must remain pure GPU; free VRAM or lower --gpu_memory_fraction.")
    mapped_gpus = _mapped_gpu_ids(model)
    if visible_gpus >= 2 and len(mapped_gpus) < 2:
        raise RuntimeError(f"Expected a multi-GPU map across {visible_gpus} visible GPUs, but got {model.hf_device_map}.")
    module_counts = Counter(str(device) for device in device_map_result.values())
    print("[model-parallel] visible_gpus=%d placement=%s max_memory=%s module_counts=%s" % (visible_gpus, device_map, max_memory, dict(sorted(module_counts.items()))))
    return model
