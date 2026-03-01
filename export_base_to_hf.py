"""Export the base (Hugging Face) PaliGemma 2 model to GGUF-ready format.

Loads the base model only (no LoRA adapters), saves the language model weights
with HuggingFace-style names (model.*) so llama.cpp convert_hf_to_gguf.py
can convert to GGUF. Does not touch or remove any fine-tuned adapter files.
"""

import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_vlm.utils import load, load_config, save_config, make_shards

LM_PREFIX = "language_model.model."


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export base PaliGemma model (no LoRA) for GGUF")
    parser.add_argument("--model", required=True, help="Path to base model (e.g. models/paligemma2-base)")
    parser.add_argument("--save-path", required=True, help="Output path for exported model")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    save_path = Path(args.save_path).resolve()

    if not model_path.is_dir():
        raise SystemExit(f"Base model not found: {model_path}")

    print("Loading base model (no adapters) …")
    model, _processor = load(str(model_path), lazy=False)
    model.eval()

    print("Collecting language model parameters …")
    all_params = dict(tree_flatten(model.parameters()))
    lm_params = {k: v for k, v in all_params.items() if k.startswith(LM_PREFIX)}
    # Only .weight and .bias (base model has no .A, .B, .original_layer)
    hf_weights = {}
    for key, arr in lm_params.items():
        if not (key.endswith(".weight") or key.endswith(".bias")):
            continue
        hf_name = key.replace(LM_PREFIX, "model.", 1)
        hf_weights[hf_name] = arr

    if not hf_weights:
        raise SystemExit("No language_model.model.* parameters found.")

    print("Loading config …")
    config = load_config(model_path)

    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving exported weights to {save_path} ({len(hf_weights)} tensors) …")

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

    print("Done. Exported base model at:", save_path)


if __name__ == "__main__":
    main()
