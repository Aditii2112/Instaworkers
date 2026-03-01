"""Fuse LoRA adapters into the base VLM and save as a single model.

Merges LoRA (A, B, original_layer) directly from the flattened parameter dict,
then saves with HuggingFace-style tensor names (model.*) so llama.cpp
convert_hf_to_gguf.py can convert to GGUF.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_vlm.utils import load, load_config, save_config, make_shards

LM_PREFIX = "language_model.model."
LORA_ORIG_RE = re.compile(r"^(.+)\.original_layer\.weight$")


def main():
    parser = argparse.ArgumentParser(description="Fuse LoRA adapters into base PaliGemma model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--de-quantize", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    adapter_path = Path(args.adapter_path).resolve()
    save_path = Path(args.save_path).resolve()

    if not model_path.is_dir():
        raise SystemExit(f"Base model not found: {model_path}")
    if not adapter_path.is_dir() or not (adapter_path / "adapter_config.json").exists():
        raise SystemExit(f"Adapter path must contain adapter_config.json: {adapter_path}")

    # Read adapter config for alpha
    with open(adapter_path / "adapter_config.json") as f:
        adapter_cfg = json.load(f)
    lora_alpha = float(adapter_cfg.get("alpha", 32.0))
    print(f"LoRA alpha = {lora_alpha}")

    print("Loading base model with LoRA adapters …")
    model, _processor = load(str(model_path), adapter_path=str(adapter_path), lazy=False)
    model.eval()

    print("Collecting parameters …")
    all_params = dict(tree_flatten(model.parameters()))
    lm_params = {k: v for k, v in all_params.items() if k.startswith(LM_PREFIX)}
    print(f"  Total params: {len(all_params)}, language_model.model.*: {len(lm_params)}")

    # Identify LoRA groups: keys ending in .original_layer.weight have siblings .A and .B
    lora_bases = {}
    for key in lm_params:
        m = LORA_ORIG_RE.match(key)
        if m:
            lora_bases[m.group(1)] = True

    print(f"  LoRA layers to merge: {len(lora_bases)}")

    # Build merged weight dict with HuggingFace names
    hf_weights = {}
    used_keys = set()

    for base_path in sorted(lora_bases):
        orig_key = f"{base_path}.original_layer.weight"
        a_key = f"{base_path}.A"
        b_key = f"{base_path}.B"

        W = lm_params[orig_key]
        A = lm_params[a_key]
        B = lm_params[b_key]

        # merged = W + alpha * (A @ B)^T  — LoRA: y = Wx + alpha * (xA)B
        # A is (in, rank), B is (rank, out); merged update is (A @ B)^T = (out, in)
        merged = W + lora_alpha * (A @ B).T

        hf_name = base_path.replace(LM_PREFIX, "model.", 1) + ".weight"
        hf_weights[hf_name] = merged
        used_keys.update([orig_key, a_key, b_key])

    # Add non-LoRA params (norms, embed, etc.)
    for key, arr in lm_params.items():
        if key in used_keys:
            continue
        if ".original_layer." in key or key.endswith(".A") or key.endswith(".B"):
            continue
        if not (key.endswith(".weight") or key.endswith(".bias")):
            continue
        hf_name = key.replace(LM_PREFIX, "model.", 1)
        hf_weights[hf_name] = arr

    if not hf_weights:
        raise SystemExit("No weights collected — something is wrong.")

    print("Loading config …")
    config = load_config(model_path)

    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving fused weights to {save_path} ({len(hf_weights)} tensors) …")

    shards = make_shards(hf_weights)
    shards_count = len(shards)
    shard_fmt = "model-{:05d}-of-{:05d}.safetensors" if shards_count > 1 else "model.safetensors"
    total_size = sum(v.nbytes for v in hf_weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    for i, shard in enumerate(shards):
        shard_name = shard_fmt.format(i + 1, shards_count) if shards_count > 1 else shard_fmt
        shard_path = save_path / shard_name
        mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})
        for key in shard:
            index_data["weight_map"][key] = shard_name

    index_data["weight_map"] = {k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])}
    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=4)

    save_config(config, save_path / "config.json")

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "spiece.model",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "generation_config.json",
    ):
        src = model_path / name
        if src.exists():
            shutil.copy2(src, save_path / name)

    print("Done. Fused model at:", save_path)


if __name__ == "__main__":
    main()
