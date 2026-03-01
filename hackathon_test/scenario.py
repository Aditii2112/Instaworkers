"""
Warehouse Safety Inspector — practice problem statement.

At the hackathon, replace this entire file with your actual problem.
The pattern stays the same:
  1. FAKE_DATA  → list of strings to seed into InstaBrain
  2. register_domain_tools(registry) → register your domain tools
  3. PROBLEM_INPUTS → dict with text_input and image_path
"""

from tools.tool_registry import ToolRegistry, ToolDefinition

# ─── Problem Statement ───────────────────────────────────────────────
#
#   "You are a warehouse safety inspector. Given an image of a warehouse
#    zone and a text concern, identify hazards, look up zone records,
#    check incident history, and file a report."
#
# ─────────────────────────────────────────────────────────────────────

# ─── Fake Data (seeded into InstaBrain for context retrieval) ────────

FAKE_DATA = [
    # Zone records
    "Zone A — Storage area, low risk. Last inspection: 2026-02-10. "
    "Status: PASSED. Contains pallets of consumer electronics. "
    "Temperature-controlled at 18°C. Fire extinguisher inspected 2026-01-15.",

    "Zone B — Loading dock, medium risk. Last inspection: 2026-01-28. "
    "Status: FLAGGED. Two incidents in the last 30 days: chemical spill "
    "on Jan 15 (cleaned, root cause: damaged container) and blocked "
    "emergency exit on Jan 22 (resolved). Forklift traffic: high.",

    "Zone C — Cold storage, high risk. Last inspection: 2026-02-05. "
    "Status: PASSED with warnings. Temperature logs show occasional "
    "spikes above -15°C threshold. Refrigeration unit serviced 2026-01-30. "
    "PPE required: insulated gloves, hard hat.",

    "Zone D — Packaging line, medium risk. Last inspection: 2026-02-01. "
    "Status: PASSED. Conveyor belt maintained on schedule. Two workers "
    "reported ergonomic discomfort (wrist strain). Ergonomic review pending.",

    # Worker schedules
    "Worker schedule Zone B — Morning shift: Raj, Priya (6am-2pm). "
    "Evening shift: Amir, Sneha (2pm-10pm). Night: unmanned, cameras only.",

    "Worker schedule Zone C — All shifts require buddy system. "
    "Morning: Chen, Fatima. Evening: Arjun, Lisa. Max 30 min continuous exposure.",

    # Safety codes
    "Safety Code SC-101: Blocked emergency exit. Severity: CRITICAL. "
    "Action: Immediate clearance required. Fine: $5,000 per occurrence.",

    "Safety Code SC-202: Chemical spill without containment. Severity: HIGH. "
    "Action: Evacuate zone, deploy spill kit, notify hazmat. Fine: $10,000.",

    "Safety Code SC-303: PPE non-compliance. Severity: MEDIUM. "
    "Action: Verbal warning first, written warning second, suspension third.",

    "Safety Code SC-404: Temperature excursion in cold storage. Severity: HIGH. "
    "Action: Check refrigeration unit, log deviation, inspect perishable goods.",

    # Equipment inventory
    "Equipment log Zone B — 2 forklifts (unit FK-07 due for service 2026-03-01, "
    "unit FK-12 serviced 2026-02-20), 1 pallet jack, 4 fire extinguishers "
    "(last checked 2026-01-15), spill kit located at dock entrance.",

    # Incident history
    "Incident INC-2026-0041: Zone B, Jan 15. Chemical spill from damaged "
    "container of industrial solvent. Area evacuated for 2 hours. "
    "Spill kit deployed. Root cause: forklift FK-07 punctured container. "
    "Corrective action: driver retrained, container stacking SOP updated.",

    "Incident INC-2026-0055: Zone B, Jan 22. Emergency exit 2B found "
    "blocked by 6 pallets during routine walkthrough. Cleared within "
    "20 minutes. Root cause: overflow from receiving area. "
    "Corrective action: max pallet limit sign posted.",
]

# ─── Test inputs (multimodal: text + image) ──────────────────────────

PROBLEM_INPUTS = {
    "text_input": (
        "Inspect Zone B loading dock. A worker reported boxes stacked "
        "near the emergency exit again and a strange smell near the "
        "chemical storage area. Check if we have any recurring violations "
        "and generate a safety report."
    ),
    "image_path": None,  # set by run_test.py after generating the image
}


# ─── Domain-Specific Tools ──────────────────────────────────────────

# In-memory "databases" the tools query against
_ZONE_DB = {
    "A": {"name": "Storage", "risk": "low", "status": "PASSED", "last_inspection": "2026-02-10"},
    "B": {"name": "Loading Dock", "risk": "medium", "status": "FLAGGED", "last_inspection": "2026-01-28"},
    "C": {"name": "Cold Storage", "risk": "high", "status": "PASSED_WITH_WARNINGS", "last_inspection": "2026-02-05"},
    "D": {"name": "Packaging Line", "risk": "medium", "status": "PASSED", "last_inspection": "2026-02-01"},
}

_INCIDENT_DB = {
    "A": [],
    "B": [
        {"id": "INC-2026-0041", "date": "2026-01-15", "type": "chemical_spill", "severity": "HIGH", "resolved": True},
        {"id": "INC-2026-0055", "date": "2026-01-22", "type": "blocked_exit", "severity": "CRITICAL", "resolved": True},
    ],
    "C": [
        {"id": "INC-2026-0033", "date": "2026-01-10", "type": "temperature_excursion", "severity": "HIGH", "resolved": True},
    ],
    "D": [],
}

_SAFETY_CODES = {
    "SC-101": {"title": "Blocked emergency exit", "severity": "CRITICAL", "fine": 5000},
    "SC-202": {"title": "Chemical spill without containment", "severity": "HIGH", "fine": 10000},
    "SC-303": {"title": "PPE non-compliance", "severity": "MEDIUM", "fine": 0},
    "SC-404": {"title": "Temperature excursion in cold storage", "severity": "HIGH", "fine": 2500},
}

_FILED_VIOLATIONS: list[dict] = []


def _lookup_zone(zone_id: str) -> dict:
    """Look up zone information by zone letter."""
    zone_id = zone_id.strip().upper()
    zone = _ZONE_DB.get(zone_id)
    if zone is None:
        return {"error": f"Unknown zone: {zone_id}", "available": list(_ZONE_DB.keys())}
    return {"zone_id": zone_id, **zone}


def _check_incidents(zone_id: str, severity: str = "") -> dict:
    """Check incident history for a zone, optionally filtered by severity."""
    zone_id = zone_id.strip().upper()
    incidents = _INCIDENT_DB.get(zone_id, [])
    if severity:
        incidents = [i for i in incidents if i["severity"].upper() == severity.upper()]
    return {"zone_id": zone_id, "incident_count": len(incidents), "incidents": incidents}


def _lookup_safety_code(code: str) -> dict:
    """Look up a safety code's details."""
    code = code.strip().upper()
    info = _SAFETY_CODES.get(code)
    if info is None:
        return {"error": f"Unknown code: {code}", "available": list(_SAFETY_CODES.keys())}
    return {"code": code, **info}


def _flag_violation(zone_id: str, code: str, description: str) -> dict:
    """File a new safety violation."""
    zone_id = zone_id.strip().upper()
    code = code.strip().upper()
    violation = {
        "zone_id": zone_id,
        "code": code,
        "description": description,
        "status": "OPEN",
    }
    _FILED_VIOLATIONS.append(violation)
    return {"filed": True, "violation": violation, "total_open": len(_FILED_VIOLATIONS)}


def _generate_report(zone_id: str, findings: str) -> dict:
    """Generate a safety inspection report for a zone."""
    zone_id = zone_id.strip().upper()
    zone = _ZONE_DB.get(zone_id, {})
    incidents = _INCIDENT_DB.get(zone_id, [])
    return {
        "report_type": "safety_inspection",
        "zone_id": zone_id,
        "zone_name": zone.get("name", "Unknown"),
        "risk_level": zone.get("risk", "unknown"),
        "current_status": zone.get("status", "unknown"),
        "past_incidents": len(incidents),
        "open_violations": len([v for v in _FILED_VIOLATIONS if v["zone_id"] == zone_id]),
        "findings": findings,
        "recommendation": "Immediate follow-up required" if zone.get("risk") in ("high", "medium") else "Routine follow-up",
    }


def register_domain_tools(registry: ToolRegistry) -> None:
    """Register all domain-specific tools. Call this before running the pipeline."""

    registry.register(ToolDefinition(
        name="lookup_zone",
        description="Look up warehouse zone info (name, risk level, inspection status) by zone letter (A/B/C/D)",
        parameters={
            "type": "object",
            "properties": {
                "zone_id": {"type": "string", "description": "Zone letter, e.g. 'B'"},
            },
            "required": ["zone_id"],
        },
        handler=_lookup_zone,
        tags=["domain", "warehouse"],
    ))

    registry.register(ToolDefinition(
        name="check_incidents",
        description="Check safety incident history for a warehouse zone. Optionally filter by severity (CRITICAL/HIGH/MEDIUM/LOW)",
        parameters={
            "type": "object",
            "properties": {
                "zone_id": {"type": "string", "description": "Zone letter, e.g. 'B'"},
                "severity": {"type": "string", "description": "Filter by severity level (optional)"},
            },
            "required": ["zone_id"],
        },
        handler=_check_incidents,
        tags=["domain", "warehouse"],
    ))

    registry.register(ToolDefinition(
        name="lookup_safety_code",
        description="Look up a safety code's title, severity, and fine amount. Codes: SC-101, SC-202, SC-303, SC-404",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Safety code, e.g. 'SC-101'"},
            },
            "required": ["code"],
        },
        handler=_lookup_safety_code,
        tags=["domain", "warehouse"],
    ))

    registry.register(ToolDefinition(
        name="flag_violation",
        description="File a new safety violation for a zone with a safety code and description",
        parameters={
            "type": "object",
            "properties": {
                "zone_id": {"type": "string", "description": "Zone letter"},
                "code": {"type": "string", "description": "Safety code (e.g. SC-101)"},
                "description": {"type": "string", "description": "Description of the violation"},
            },
            "required": ["zone_id", "code", "description"],
        },
        handler=_flag_violation,
        requires_approval=True,
        tags=["domain", "warehouse"],
    ))

    registry.register(ToolDefinition(
        name="generate_report",
        description="Generate a safety inspection report summarizing zone status, incidents, and findings",
        parameters={
            "type": "object",
            "properties": {
                "zone_id": {"type": "string", "description": "Zone letter"},
                "findings": {"type": "string", "description": "Inspector's findings summary"},
            },
            "required": ["zone_id", "findings"],
        },
        handler=_generate_report,
        tags=["domain", "warehouse"],
    ))
