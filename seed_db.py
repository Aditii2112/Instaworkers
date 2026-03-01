#!/usr/bin/env python3
"""Seed inventory.db with warehouse data for Sovereign Sentinel.

Run: python seed_db.py
Creates data/inventory.db with SKUs like Chemical-X, Grade-A Produce, etc.
"""

import sqlite3
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent
INVENTORY_DB = ROOT_DIR / "data" / "inventory.db"


def seed_inventory():
    Path(INVENTORY_DB).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(INVENTORY_DB))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku TEXT NOT NULL,
            location TEXT NOT NULL,
            product_name TEXT,
            status TEXT DEFAULT 'ACTIVE',
            quantity INTEGER DEFAULT 0,
            last_updated REAL
        )
    """)

    cur.execute("DELETE FROM inventory")

    now = time.time()
    sample_data = [
        ("SKU-CX01", "Aisle_1", "Chemical-X (Industrial Solvent)", "ACTIVE", 120),
        ("SKU-CX02", "Aisle_1", "Chemical-X (Cleaning Agent)", "ACTIVE", 85),
        ("SKU-GA01", "Aisle_2", "Grade-A Produce (Organic Lettuce)", "ACTIVE", 200),
        ("SKU-GA02", "Aisle_2", "Grade-A Produce (Tomatoes)", "ACTIVE", 150),
        ("SKU-GA03", "Aisle_2", "Grade-A Produce (Avocados)", "ACTIVE", 90),
        ("SKU-FZ01", "Cold_Storage", "Frozen Grade-A Poultry", "ACTIVE", 300),
        ("SKU-FZ02", "Cold_Storage", "Frozen Seafood Premium", "ACTIVE", 180),
        ("SKU-DY01", "Aisle_3", "Dairy Products (Milk)", "ACTIVE", 250),
        ("SKU-DY02", "Aisle_3", "Dairy Products (Cheese)", "ACTIVE", 100),
        ("SKU-PH01", "Aisle_4", "Pharmaceuticals (First Aid)", "ACTIVE", 75),
        ("SKU-PH02", "Aisle_4", "Pharmaceuticals (Disinfectant)", "ACTIVE", 60),
        ("SKU-HZ01", "Loading_Dock", "Hazmat Incoming Shipment", "ACTIVE", 0),
        ("SKU-HZ02", "Loading_Dock", "Industrial Lubricant Drums", "ACTIVE", 40),
        ("SKU-EL01", "Rack_04", "Electronics (Sensors)", "ACTIVE", 500),
        ("SKU-EL02", "Rack_04", "Electronics (Wiring Harnesses)", "ACTIVE", 200),
        ("SKU-PKG01", "Aisle_5", "Packaging Materials (Foam)", "ACTIVE", 1000),
        ("SKU-PKG02", "Aisle_5", "Packaging Materials (Bubble Wrap)", "ACTIVE", 800),
        ("SKU-CL01", "Warehouse", "Cleaning Supplies (General)", "ACTIVE", 150),
        ("SKU-DMG01", "Aisle_4", "Damaged Goods - Pending Review", "QUARANTINED", 5),
        ("SKU-SP01", "Aisle_2", "Spice Rack (Grade-A)", "ACTIVE", 45),
    ]

    for sku, location, product_name, status, qty in sample_data:
        cur.execute(
            "INSERT INTO inventory (sku, location, product_name, status, quantity, last_updated) VALUES (?, ?, ?, ?, ?, ?)",
            (sku, location, product_name, status, qty, now),
        )

    conn.commit()
    conn.close()
    print(f"[DB] Seeded {len(sample_data)} SKUs into {INVENTORY_DB}")


if __name__ == "__main__":
    seed_inventory()
