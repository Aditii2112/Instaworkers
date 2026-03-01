"""Sovereign Sentinel — Data generation + MLX LoRA fine-tuning pipeline.

Generates train.jsonl from three local datasets, then kicks off
mlx_lm.lora fine-tuning on PaliGemma 2 (3B).  100 % offline — no
HuggingFace Hub downloads.

Datasets (expected directory layout under ./datasets/):
  datasets/
    package_damage/          # keremberke/package-damage-detection
      train/images/
      train/labels/
    ip102/                   # IP102 pest dataset
      train/
        class_<id>/          # one sub-dir per class
    mvtec/                   # MVTec anomaly detection
      <category>/
        train/good/
        test/<defect_type>/
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
MODEL_DIR = ROOT / "models" / "paligemma2-base"
ADAPTERS_DIR = ROOT / "adapters"
TRAIN_JSONL = ROOT / "train.jsonl"

# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------

DAMAGE_LABELS = {
    0: "dent",
    1: "tear",
    2: "crush",
    3: "water_damage",
    4: "puncture",
}

IP102_LABELS = {
    0: "rice_leafhopper", 1: "brown_planthopper", 2: "white_backed_planthopper",
    3: "small_brown_planthopper", 4: "rice_water_weevil", 5: "rice_gall_midge",
    6: "rice_stem_borer", 7: "asiatic_rice_borer", 8: "yellow_rice_borer",
    9: "rice_shell_pest", 10: "grain_moth", 11: "rice_leaf_roller",
    12: "rice_leaf_caterpillar", 13: "paddy_stem_maggot",
}

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]

# ---------------------------------------------------------------------------
# Prompt templates (SEE / REASON / ACT)
# ---------------------------------------------------------------------------

QUERY_VARIANTS = [
    "Analyze this warehouse feed.",
    "Inspect this image for defects.",
    "What anomalies do you detect?",
    "Evaluate this item for quality issues.",
    "Perform a safety inspection on this frame.",
]


def _fmt_record(image_path: str, see: str, reason: str, act: dict) -> dict:
    """Build a single training record.

    Produces two formats:
      - "text"     : flat string for train.jsonl reference copy
      - "messages" : chat-turns list required by mlx_vlm.lora
      - "images"   : list of image paths required by mlx_vlm.lora
    """
    query = random.choice(QUERY_VARIANTS)
    act_str = json.dumps([act], separators=(",", ":"))
    assistant_text = f"SEE: {see} REASON: {reason} ACT: {act_str}"
    text = (
        f"user: <image>\n{query}\n"
        f"assistant: {assistant_text}"
    )
    return {
        "text": text,
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": assistant_text},
        ],
        "images": str(image_path),
    }


# ---------------------------------------------------------------------------
# Dataset processors
# ---------------------------------------------------------------------------

def process_package_damage(base: Path) -> list[dict]:
    """keremberke/package-damage-detection  (YOLO-format labels)."""
    records = []
    images_dir = base / "train" / "images"
    labels_dir = base / "train" / "labels"
    if not images_dir.exists():
        print(f"  [SKIP] {images_dir} not found")
        return records

    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_file = labels_dir / img.with_suffix(".txt").name
        if not label_file.exists():
            continue
        with open(label_file) as fh:
            lines = fh.read().strip().splitlines()
        if not lines:
            continue

        class_id = int(lines[0].split()[0])
        label = DAMAGE_LABELS.get(class_id, "unknown_damage")

        records.append(_fmt_record(
            image_path=img,
            see=f"[Package damage detected: {label}]",
            reason=f"Package shows signs of {label}; flagging for quarantine and reporting incident to warehouse supervisor.",
            act={"tool": "flag_violation", "params": {"type": "package_damage", "severity": "high", "label": label}},
        ))
    return records


def process_ip102(base: Path) -> list[dict]:
    """IP102 pest classification dataset (folder-per-class)."""
    records = []
    train_dir = base / "train"
    if not train_dir.exists():
        print(f"  [SKIP] {train_dir} not found")
        return records

    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        try:
            class_id = int(class_dir.name.split("_")[-1])
        except ValueError:
            class_id = -1
        label = IP102_LABELS.get(class_id, class_dir.name)

        for img in sorted(class_dir.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            records.append(_fmt_record(
                image_path=img,
                see=f"[Pest detected: {label}]",
                reason=f"Identified {label} infestation risk in stored goods area. Recommend pest-control dispatch and zone isolation.",
                act={"tool": "generate_report", "params": {"type": "pest_alert", "pest": label, "action": "dispatch_pest_control"}},
            ))
    return records


def process_mvtec(base: Path) -> list[dict]:
    """MVTec anomaly detection — good vs defect sub-folders."""
    records = []
    if not base.exists():
        print(f"  [SKIP] {base} not found")
        return records

    for category in MVTEC_CATEGORIES:
        cat_dir = base / category

        good_dir = cat_dir / "train" / "good"
        if good_dir.exists():
            for img in sorted(good_dir.iterdir()):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                records.append(_fmt_record(
                    image_path=img,
                    see=f"[{category}: no defect]",
                    reason=f"{category.title()} passes visual quality check — no anomalies observed.",
                    act={"tool": "check_incidents", "params": {"zone": "quality_line", "status": "pass", "item": category}},
                ))

        test_dir = cat_dir / "test"
        if not test_dir.exists():
            continue
        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir() or defect_dir.name == "good":
                continue
            defect_type = defect_dir.name
            for img in sorted(defect_dir.iterdir()):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                records.append(_fmt_record(
                    image_path=img,
                    see=f"[{category} defect: {defect_type}]",
                    reason=f"Anomaly detected on {category} — {defect_type}. Pulling item from line and logging defect.",
                    act={"tool": "flag_violation", "params": {"type": "quality_defect", "category": category, "defect": defect_type}},
                ))
    return records


# ---------------------------------------------------------------------------
# JSONL writer + train/valid split
# ---------------------------------------------------------------------------

def write_splits(records: list[dict], out_dir: Path, val_frac: float = 0.1):
    """Shuffle and write train.jsonl / valid.jsonl for mlx_lm.lora."""
    random.shuffle(records)
    split_idx = max(1, int(len(records) * (1 - val_frac)))
    train_set = records[:split_idx]
    valid_set = records[split_idx:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"

    for path, subset in [(train_path, train_set), (valid_path, valid_set)]:
        with open(path, "w") as fh:
            for rec in subset:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(subset):,} records → {path}")

    flat_path = ROOT / "train.jsonl"
    with open(flat_path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):,} total records → {flat_path}")

    return train_path, valid_path


# ---------------------------------------------------------------------------
# MLX LoRA fine-tuning
# ---------------------------------------------------------------------------

def _find_resume_adapter_path(adapter_dir: Path) -> Path | None:
    """Return path to latest saved adapter so we can resume, or None."""
    config_file = adapter_dir / "adapter_config.json"
    if config_file.exists():
        return adapter_dir
    parent = adapter_dir.parent
    stem = adapter_dir.name
    candidates = sorted(
        (p for p in parent.glob(f"epoch_*_{stem}") if (p / "adapter_config.json").exists()),
        key=lambda p: int(p.name.split("_")[1]) if len(p.name.split("_")) > 2 and p.name.split("_")[1].isdigit() else -1,
        reverse=True,
    )
    return candidates[0] if candidates else None


def run_lora_training(
    data_dir: Path,
    model_dir: Path,
    adapter_dir: Path,
    resume: bool = False,
):
    """Launch mlx_vlm.lora (vision-language LoRA) sized for 24 GB Apple Silicon.

    Uses --steps 0 so mlx_vlm sets steps = len(dataset) // batch_size per epoch.
    With --save-after-epoch, each epoch saves a checkpoint; use --resume to continue.
    """
    adapter_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "train.jsonl"
    num_samples = 0
    if train_file.exists():
        with open(train_file) as f:
            num_samples = sum(1 for _ in f)
    steps_per_epoch = max(1, num_samples // 1)
    steps_arg = "0"

    cmd = [
        sys.executable, "-m", "mlx_vlm.lora",
        "--model-path", str(model_dir),
        "--dataset", str(data_dir),
        "--output-path", str(adapter_dir),
        "--lora-rank", "16",
        "--lora-alpha", "32",
        "--lora-dropout", "0.05",
        "--learning-rate", "1e-4",
        "--batch-size", "1",
        "--epochs", "1",
        "--steps", steps_arg,
        "--print-every", "25",
        "--image-resize-shape", "224", "224",
        "--save-after-epoch",
    ]

    adapter_path = None
    if resume:
        adapter_path = _find_resume_adapter_path(adapter_dir)
        if adapter_path is None:
            print("⚠ --resume set but no checkpoint found; starting from scratch.")
        else:
            cmd.extend(["--adapter-path", str(adapter_path)])
            print("  Resuming from:", adapter_path)

    print("\n" + "=" * 70)
    print("  MLX VLM LoRA Fine-Tuning — Sovereign Sentinel")
    print("=" * 70)
    print(f"  Model   : {model_dir}")
    print(f"  Data    : {data_dir}")
    print(f"  Adapters: {adapter_dir}")
    print(f"  Samples : {num_samples:,} → steps/epoch = {steps_per_epoch} (steps=0 = auto)")
    print(f"  Config  : rank=16, alpha=32, epochs=1, bs=1, save-after-epoch=on")
    if adapter_path:
        print(f"  Resume  : {adapter_path}")
    print("=" * 70 + "\n")

    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sovereign Sentinel — data generation + MLX LoRA fine-tuning"
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        help="Only generate train.jsonl; skip training.",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only run training (data must already exist).",
    )
    parser.add_argument(
        "--datasets-dir", type=Path, default=DATASETS_DIR,
        help="Root directory containing dataset sub-folders.",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=MODEL_DIR,
        help="Path to the base PaliGemma 2 weights.",
    )
    parser.add_argument(
        "--adapter-dir", type=Path, default=ADAPTERS_DIR,
        help="Output directory for LoRA adapter weights.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last saved checkpoint (adapters or epoch_*_adapters).",
    )
    args = parser.parse_args()
    random.seed(args.seed)

    data_out_dir = ROOT / "finetune_data"

    # ── Data generation ──────────────────────────────────────────────────
    if not args.train_only:
        print("\n▸ Generating training data …")
        all_records: list[dict] = []

        print("  ┌ package_damage …")
        all_records.extend(process_package_damage(args.datasets_dir / "package_damage"))

        print("  ├ ip102 …")
        all_records.extend(process_ip102(args.datasets_dir / "ip102"))

        print("  └ mvtec …")
        all_records.extend(process_mvtec(args.datasets_dir / "mvtec"))

        if not all_records:
            print("\n⚠  No images found. Make sure datasets are under:", args.datasets_dir)
            print("   Expected layout:")
            print("     datasets/package_damage/train/images/")
            print("     datasets/ip102/train/class_<id>/")
            print("     datasets/mvtec/<category>/train/good/")
            sys.exit(1)

        print(f"\n  Total records: {len(all_records):,}")
        write_splits(all_records, data_out_dir)

    # ── Training ─────────────────────────────────────────────────────────
    if not args.generate_only:
        if not (data_out_dir / "train.jsonl").exists():
            print("✗  train.jsonl not found — run data generation first.")
            sys.exit(1)
        if not args.model_dir.exists():
            print(f"✗  Model directory not found: {args.model_dir}")
            print("   Download google/paligemma2-3b-pt-224 into that path first.")
            sys.exit(1)

        run_lora_training(
            data_out_dir,
            args.model_dir,
            args.adapter_dir,
            resume=args.resume,
        )
        print("\n✓  Training complete. Adapters saved to:", args.adapter_dir)
        print("   Next step → ./deploy_to_ollama.sh")


if __name__ == "__main__":
    main()
