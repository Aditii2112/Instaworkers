"""Generate a synthetic warehouse floor image for multimodal testing.

Creates a realistic-looking top-down warehouse layout with:
- Colored zones (A, B, C, D)
- Hazard indicators in Zone B (boxes near exit, chemical warning)
- Labels and markers the vision pipeline can pick up

At hackathon, replace this with actual input images for your problem.
"""

import cv2
import numpy as np
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


def generate_warehouse_image(output_path: str | None = None) -> str:
    """Generate a synthetic warehouse floor image and return the file path."""
    if output_path is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(DATA_DIR / "test_warehouse.png")

    img = np.ones((800, 1200, 3), dtype=np.uint8) * 240  # light gray bg

    # Zone A — top-left (green = safe)
    cv2.rectangle(img, (20, 20), (580, 380), (200, 235, 200), -1)
    cv2.rectangle(img, (20, 20), (580, 380), (80, 160, 80), 2)
    cv2.putText(img, "ZONE A - Storage", (180, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 100, 40), 2)
    cv2.putText(img, "STATUS: PASSED", (220, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 100, 40), 1)
    # Neat pallet rows
    for row in range(3):
        for col in range(6):
            x = 60 + col * 85
            y = 140 + row * 75
            cv2.rectangle(img, (x, y), (x + 60, y + 50), (160, 180, 160), -1)
            cv2.rectangle(img, (x, y), (x + 60, y + 50), (100, 130, 100), 1)

    # Zone B — top-right (orange = flagged, this is the problem zone)
    cv2.rectangle(img, (620, 20), (1180, 380), (200, 220, 240), -1)
    cv2.rectangle(img, (620, 20), (1180, 380), (40, 80, 200), 2)
    cv2.putText(img, "ZONE B - Loading Dock", (720, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 50, 180), 2)
    cv2.putText(img, "STATUS: FLAGGED", (770, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

    # Emergency exit (red door marker)
    cv2.rectangle(img, (1100, 150), (1170, 250), (60, 60, 200), -1)
    cv2.putText(img, "EXIT", (1110, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, "2B", (1120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Boxes blocking the exit (the hazard!)
    for i in range(5):
        x = 1030 + (i % 3) * 30
        y = 160 + (i // 3) * 50
        cv2.rectangle(img, (x, y), (x + 25, y + 40), (80, 130, 200), -1)
        cv2.rectangle(img, (x, y), (x + 25, y + 40), (40, 60, 120), 2)
    cv2.putText(img, "!! BLOCKED !!", (1020, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Chemical storage area with warning triangle
    cv2.rectangle(img, (640, 250), (780, 360), (180, 210, 240), -1)
    cv2.rectangle(img, (640, 250), (780, 360), (0, 0, 180), 2)
    cv2.putText(img, "CHEM", (670, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180), 2)
    # Warning triangle
    tri = np.array([[710, 320], [690, 355], [730, 355]], np.int32)
    cv2.fillPoly(img, [tri], (0, 200, 255))
    cv2.putText(img, "!", (704, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Forklift icon (simple)
    cv2.rectangle(img, (850, 200), (920, 280), (100, 100, 100), -1)
    cv2.circle(img, (860, 290), 12, (60, 60, 60), -1)
    cv2.circle(img, (910, 290), 12, (60, 60, 60), -1)
    cv2.putText(img, "FK-07", (850, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)

    # Zone C — bottom-left (blue = cold storage)
    cv2.rectangle(img, (20, 420), (580, 780), (230, 220, 200), -1)
    cv2.rectangle(img, (20, 420), (580, 780), (180, 120, 60), 2)
    cv2.putText(img, "ZONE C - Cold Storage", (140, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 80, 30), 2)
    cv2.putText(img, "STATUS: WARNINGS", (200, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 80, 30), 1)
    cv2.putText(img, "-18 C", (260, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 120, 60), 3)
    # Shelving
    for row in range(3):
        y = 580 + row * 60
        cv2.rectangle(img, (60, y), (540, y + 40), (210, 200, 180), -1)
        cv2.rectangle(img, (60, y), (540, y + 40), (160, 140, 120), 1)

    # Zone D — bottom-right (yellow = packaging)
    cv2.rectangle(img, (620, 420), (1180, 780), (210, 230, 240), -1)
    cv2.rectangle(img, (620, 420), (1180, 780), (50, 150, 200), 2)
    cv2.putText(img, "ZONE D - Packaging Line", (710, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 120, 170), 2)
    cv2.putText(img, "STATUS: PASSED", (780, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 120, 170), 1)
    # Conveyor belt
    cv2.rectangle(img, (660, 540), (1140, 570), (140, 170, 200), -1)
    for x in range(660, 1140, 40):
        cv2.line(img, (x, 540), (x + 20, 570), (100, 130, 160), 1)
    cv2.putText(img, ">>> CONVEYOR >>>", (780, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 100, 140), 1)

    # Workers (stick figures) in Zone B
    for px, py in [(780, 170), (920, 160)]:
        cv2.circle(img, (px, py), 10, (50, 50, 50), -1)       # head
        cv2.line(img, (px, py + 10), (px, py + 40), (50, 50, 50), 2)  # body
        cv2.line(img, (px - 12, py + 25), (px + 12, py + 25), (50, 50, 50), 2)  # arms
        cv2.line(img, (px, py + 40), (px - 10, py + 60), (50, 50, 50), 2)  # legs
        cv2.line(img, (px, py + 40), (px + 10, py + 60), (50, 50, 50), 2)

    cv2.imwrite(output_path, img)
    return output_path


if __name__ == "__main__":
    path = generate_warehouse_image()
    print(f"Test image saved to: {path}")
