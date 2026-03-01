"""Inventory Agent — queries inventory.db for SKUs at a visual location."""

import sqlite3
from pathlib import Path

from config import ROOT_DIR

INVENTORY_DB = ROOT_DIR / "data" / "inventory.db"


def query_skus_at_location(location: str) -> list[dict]:
    """Query inventory.db for SKUs at the given location.
    Returns list of {sku, product_name, status, quantity}.
    """
    if not INVENTORY_DB.exists():
        return []

    conn = sqlite3.connect(str(INVENTORY_DB))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT sku, product_name, status, quantity FROM inventory WHERE location = ?",
        (location,),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]
