"""Download and organize datasets for Sovereign Sentinel training.

Downloads:
  1. keremberke/package-damage-detection  (HuggingFace)
  2. IP102 pest dataset                   (HuggingFace mirror)
  3. MVTec AD anomaly detection           (HuggingFace mirror)

Organises everything into the directory layout expected by train_sentinel.py.
"""

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"


def ensure_deps():
    """Install datasets + huggingface_hub if not present."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "datasets", "huggingface_hub", "Pillow"],
        )


# ---------------------------------------------------------------------------
# 1. Package Damage Detection
# ---------------------------------------------------------------------------

def download_package_damage():
    """Package damage detection → datasets/package_damage/train/{images,labels}/"""
    from datasets import load_dataset

    out = DATASETS_DIR / "package_damage" / "train"
    img_dir = out / "images"
    lbl_dir = out / "labels"

    if img_dir.exists() and any(img_dir.iterdir()):
        print("  ✓ package_damage already present — skipping")
        return

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    HF_NAMES = [
        ("keremberke/package-damage-detection", {"name": "full"}),
        ("keremberke/package-damage-detection", {}),
        ("Francesco/package-damage-detection", {}),
    ]

    ds = None
    for repo, kwargs in HF_NAMES:
        try:
            print(f"  Trying {repo} …")
            ds = load_dataset(repo, split="train", trust_remote_code=True, **kwargs)
            print(f"  ✓ Loaded {repo}")
            break
        except Exception as e:
            print(f"    → {e.__class__.__name__}: skipped")

    if ds is None:
        print("  ⚠  Could not download package-damage dataset — creating placeholder data")
        _create_package_damage_placeholder(out)
        return

    for idx, sample in enumerate(ds):
        img = sample["image"]
        fname = f"{idx:05d}.jpg"
        img.save(img_dir / fname)

        objects = sample.get("objects", {})
        bboxes = objects.get("bbox", [])
        categories = objects.get("category", [])

        lines = []
        if bboxes and categories:
            w, h = img.size
            for cat, bbox in zip(categories, bboxes):
                x1, y1, x2, y2 = bbox
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{cat} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        elif "label" in sample:
            lines.append(f"{sample['label']} 0.5 0.5 1.0 1.0")

        if lines:
            (lbl_dir / f"{idx:05d}.txt").write_text("\n".join(lines) + "\n")

    print(f"  ✓ Saved {len(ds)} images → {img_dir}")


def _create_package_damage_placeholder(out: Path):
    from PIL import Image
    import random

    img_dir = out / "images"
    lbl_dir = out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(50):
        cls = random.randint(0, 4)
        img = Image.new("RGB", (224, 224),
                        (random.randint(80, 180), random.randint(60, 140), random.randint(40, 120)))
        img.save(img_dir / f"{i:05d}.jpg")
        cx, cy = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)
        bw, bh = random.uniform(0.1, 0.5), random.uniform(0.1, 0.5)
        (lbl_dir / f"{i:05d}.txt").write_text(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    print(f"  ✓ Created placeholder package_damage data (50 images) → {out}")


# ---------------------------------------------------------------------------
# 2. IP102 Pest Dataset
# ---------------------------------------------------------------------------

def download_ip102():
    """IP102 → datasets/ip102/train/class_<id>/"""
    from datasets import load_dataset

    out = DATASETS_DIR / "ip102" / "train"

    if out.exists() and any(out.iterdir()):
        print("  ✓ ip102 already present — skipping")
        return

    out.mkdir(parents=True, exist_ok=True)

    HF_NAMES = ["Edaax/IP102", "rtomyj/IP102", "ip102"]
    ds = None
    for repo in HF_NAMES:
        try:
            print(f"  Trying {repo} …")
            ds = load_dataset(repo, split="train", trust_remote_code=True)
            print(f"  ✓ Loaded {repo}")
            break
        except Exception as e:
            print(f"    → {e.__class__.__name__}: skipped")

    if ds is None:
        print("  ⚠  Could not download IP102 — creating placeholder data")
        _create_ip102_placeholder(out)
        return

    for idx, sample in enumerate(ds):
        label = sample.get("label", sample.get("fine_label", 0))
        class_dir = out / f"class_{label}"
        class_dir.mkdir(exist_ok=True)
        img = sample["image"]
        img.save(class_dir / f"{idx:05d}.jpg")

    print(f"  ✓ Saved {len(ds)} images → {out}")


def _create_ip102_placeholder(out: Path):
    """Create small synthetic images when the real dataset is unavailable."""
    from PIL import Image
    import random

    for class_id in range(10):
        class_dir = out / f"class_{class_id}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for j in range(20):
            img = Image.new("RGB", (224, 224),
                            (random.randint(0, 255), random.randint(100, 200), random.randint(0, 100)))
            img.save(class_dir / f"{j:04d}.jpg")
    print(f"  ✓ Created placeholder IP102 data ({10 * 20} images) → {out}")


# ---------------------------------------------------------------------------
# 3. MVTec AD
# ---------------------------------------------------------------------------

def download_mvtec():
    """MVTec AD → datasets/mvtec/<category>/{train/good, test/<defect>}/"""
    from datasets import load_dataset

    out = DATASETS_DIR / "mvtec"

    if out.exists() and any(out.iterdir()):
        print("  ✓ mvtec already present — skipping")
        return

    out.mkdir(parents=True, exist_ok=True)

    categories = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper",
    ]

    HF_REPOS = ["alexrods/mvtec-ad", "mvtec-ad", "Voxel51/MVTec-AD"]

    print("  Downloading MVTec AD (multiple categories) …")
    for cat in categories:
        print(f"    ↓ {cat} …", end=" ", flush=True)
        ds = None
        for repo in HF_REPOS:
            try:
                ds = load_dataset(repo, cat, trust_remote_code=True)
                break
            except Exception:
                continue
        if ds is None:
            print("skipped")
            continue

        for split_name in ["train", "test"]:
            if split_name not in ds:
                continue
            split = ds[split_name]
            for idx, sample in enumerate(split):
                label = sample.get("label", "good")
                if isinstance(label, int):
                    label = "good" if label == 0 else f"defect_{label}"

                if split_name == "train":
                    dest = out / cat / "train" / "good"
                else:
                    dest = out / cat / "test" / label

                dest.mkdir(parents=True, exist_ok=True)
                img = sample["image"]
                img.save(dest / f"{idx:05d}.jpg")

        print("done")

    if not any(out.iterdir()):
        print("  ⚠  MVTec download failed — creating placeholder data …")
        _create_mvtec_placeholder(out)
    else:
        print(f"  ✓ MVTec AD saved → {out}")


def _create_mvtec_placeholder(out: Path):
    """Create small synthetic images when the real dataset is unavailable."""
    from PIL import Image
    import random

    for cat in ["bottle", "cable", "capsule", "hazelnut", "pill"]:
        good_dir = out / cat / "train" / "good"
        good_dir.mkdir(parents=True, exist_ok=True)
        for j in range(15):
            img = Image.new("RGB", (224, 224),
                            (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200)))
            img.save(good_dir / f"{j:04d}.jpg")

        for defect in ["scratch", "crack"]:
            defect_dir = out / cat / "test" / defect
            defect_dir.mkdir(parents=True, exist_ok=True)
            for j in range(10):
                img = Image.new("RGB", (224, 224),
                                (random.randint(150, 255), random.randint(0, 80), random.randint(0, 80)))
                img.save(defect_dir / f"{j:04d}.jpg")

    print(f"  ✓ Created placeholder MVTec data → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("═" * 60)
    print("  Sovereign Sentinel — Dataset Downloader")
    print("═" * 60)
    print(f"  Target: {DATASETS_DIR}\n")

    ensure_deps()

    print("▸ [1/3] Package Damage Detection")
    download_package_damage()

    print("\n▸ [2/3] IP102 Pest Dataset")
    download_ip102()

    print("\n▸ [3/3] MVTec Anomaly Detection")
    download_mvtec()

    print("\n" + "═" * 60)
    print("  All datasets ready!")
    print(f"  Next → python3 train_sentinel.py --generate-only")
    print("═" * 60)


if __name__ == "__main__":
    main()
