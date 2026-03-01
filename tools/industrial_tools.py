"""Industrial tools for the Sovereign Sentinel Facility Intelligence System.

dispatch_visual_ticket — Route to Facilities (leak), Inventory (spoilage), Pest Control (pest).
                         Captures the exact anomaly frame as .jpg and generates ticket.json.
quarantine_inventory_sku — SQL UPDATE on inventory.db to set status=QUARANTINED.
broadcast_safety_alert — MacOS `say` command to announce hazard and dispatched team.
"""

import json
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path

import cv2

from config import ROOT_DIR

ALERTS_DIR = ROOT_DIR / "data" / "alerts"
INVENTORY_DB = ROOT_DIR / "data" / "inventory.db"

TEAM_ROUTING = {
    "leak": "Facilities",
    "spoilage": "Inventory",
    "pest": "Pest Control",
    "spill": "Facilities",
    "damage": "Facilities",
    "hazard": "Facilities",
    "fire": "Facilities",
    "smoke": "Facilities",
}


def dispatch_visual_ticket(
    anomaly: str,
    severity: str,
    location: str,
    image_path: str | None = None,
    video_path: str | None = None,
    frame_index: int | None = None,
    team: str | None = None,
) -> dict:
    """Route anomaly to the correct team, capture the exact video frame, save ticket.

    If video_path and frame_index are provided, extracts that specific frame.
    Otherwise falls back to image_path (pre-extracted frame).
    """
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    anomaly_lower = (anomaly or "unknown").lower()
    assigned_team = team or TEAM_ROUTING.get(anomaly_lower, "Facilities")

    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    frame_name = f"{ticket_id}_frame.jpg"
    frame_path = ALERTS_DIR / frame_name
    frame_saved = False

    if video_path and frame_index is not None:
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            if ret:
                cv2.imwrite(str(frame_path), frame)
                frame_saved = True
        except Exception:
            pass

    if not frame_saved and image_path and Path(image_path).exists():
        import shutil
        shutil.copy2(image_path, frame_path)
        frame_saved = True

    if not frame_saved:
        frame_path = None

    ticket = {
        "ticket_id": ticket_id,
        "anomaly": anomaly,
        "severity": severity,
        "location": location,
        "team": assigned_team,
        "image_path": str(frame_path) if frame_path else None,
        "image_filename": frame_name if frame_path else None,
        "status": "OPEN",
        "created_at": time.time(),
    }

    ticket_path = ALERTS_DIR / f"{ticket_id}_ticket.json"
    with open(ticket_path, "w") as f:
        json.dump(ticket, f, indent=2)

    return {
        "ticket_id": ticket_id,
        "team": assigned_team,
        "severity": severity,
        "location": location,
        "anomaly": anomaly,
        "image_saved": str(frame_path) if frame_path else None,
        "ticket_path": str(ticket_path),
    }


def quarantine_inventory_sku(sku: str, location: str | None = None) -> dict:
    """Execute SQL UPDATE on inventory.db to set status=QUARANTINED for the SKU."""
    if not INVENTORY_DB.exists():
        return {"error": "inventory.db not found", "status": "error"}

    conn = sqlite3.connect(str(INVENTORY_DB))
    cur = conn.cursor()

    if location:
        cur.execute(
            "UPDATE inventory SET status = 'QUARANTINED', last_updated = ? WHERE sku = ? AND location = ?",
            (time.time(), sku, location),
        )
    else:
        cur.execute(
            "UPDATE inventory SET status = 'QUARANTINED', last_updated = ? WHERE sku = ?",
            (time.time(), sku),
        )

    rows = cur.rowcount
    conn.commit()

    cur.execute(
        "SELECT sku, product_name, location, status FROM inventory WHERE sku = ?",
        (sku,),
    )
    updated = [dict(zip(["sku", "product_name", "location", "status"], r)) for r in cur.fetchall()]
    conn.close()

    return {
        "sku": sku,
        "location": location,
        "status": "QUARANTINED",
        "rows_updated": rows,
        "updated_records": updated,
    }


def broadcast_safety_alert(location: str, team: str, message: str | None = None) -> dict:
    """Use MacOS say command to announce the hazard and dispatched team."""
    announcement = message or f"Attention. Hazard detected in {location}. {team} team has been dispatched. Please clear the area."
    try:
        subprocess.Popen(
            ["say", "-v", "Samantha", announcement],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return {"announced": True, "message": announcement, "team": team, "location": location}
    except (FileNotFoundError, OSError) as e:
        return {"announced": False, "error": str(e), "message": announcement}
