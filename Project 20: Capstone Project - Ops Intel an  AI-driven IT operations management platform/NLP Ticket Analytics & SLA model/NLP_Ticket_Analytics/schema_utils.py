"""
schema_utils.py
───────────────
Schema-aware data pipeline utilities for the IT Ops ticket analytics stream.

Purpose
-------
Detect and handle schema changes (added columns, removed columns, type changes,
renamed columns) across any of the 9 source tables in IT_Ops_Intern_Ready_*.xlsx.

Usage
-----
  from schema_utils import (
      load_schema_registry, validate_and_diff,
      dynamic_feature_pools, SchemaChangeError
  )

All 4 notebooks import this module at startup.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
# The registry defines what columns are EXPECTED in each table.
# required=True  → missing column raises SchemaChangeError (pipeline stops)
# required=False → missing column is logged as a WARNING only
# New columns found at runtime are logged as INFO and auto-classified.
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_REGISTRY = {

    "service_tickets": {
        "ticket_id":                   {"dtype": "str",   "required": True},
        "client_id":                   {"dtype": "str",   "required": True},
        "asset_id":                    {"dtype": "str",   "required": True},
        "engineer_id":                 {"dtype": "str",   "required": True},
        "ticket_created_timestamp":    {"dtype": "datetime", "required": True},
        "ticket_close_timestamp":      {"dtype": "datetime", "required": False},
        "ticket_category":             {"dtype": "cat",   "required": True},
        "issue_type":                  {"dtype": "cat",   "required": True},
        "ticket_priority":             {"dtype": "cat",   "required": True},
        "ticket_status":               {"dtype": "cat",   "required": False},
        "ticket_channel":              {"dtype": "cat",   "required": False},
        "escalation_flag":             {"dtype": "num",   "required": False},
        "ticket_reopen_flag":          {"dtype": "num",   "required": False},
        "first_response_time_minutes": {"dtype": "num",   "required": True},
        "resolution_time_minutes":     {"dtype": "num",   "required": True},
        "sla_breach_flag":             {"dtype": "num",   "required": True},
        "ticket_description":          {"dtype": "text",  "required": False},
        "resolution_notes":            {"dtype": "text",  "required": False},
        "ticket_detailed_description": {"dtype": "text",  "required": False},  # NLP primary field
    },

    "engineer_master": {
        "engineer_id":      {"dtype": "str",  "required": True},
        "engineer_name":    {"dtype": "str",  "required": True},
        "specialization":   {"dtype": "cat",  "required": True},
        "support_level":    {"dtype": "cat",  "required": True},
        "experience_years": {"dtype": "num",  "required": True},
        "shift_type":       {"dtype": "cat",  "required": True},
        "active_year":      {"dtype": "num",  "required": False},
    },

    "client_master": {
        "client_id":           {"dtype": "str",      "required": True},
        "client_name":         {"dtype": "str",      "required": False},
        "industry_sector":     {"dtype": "cat",      "required": True},
        "region":              {"dtype": "cat",      "required": True},
        "service_tier":        {"dtype": "cat",      "required": True},
        "contract_start_date": {"dtype": "datetime", "required": False},
        "contract_end_date":   {"dtype": "datetime", "required": False},
        "active_year":         {"dtype": "num",      "required": False},
    },

    "asset_master": {
        "asset_id":           {"dtype": "str",      "required": True},
        "client_id":          {"dtype": "str",      "required": True},
        "device_type":        {"dtype": "cat",      "required": True},
        "manufacturer":       {"dtype": "cat",      "required": False},
        "model_number":       {"dtype": "str",      "required": False},
        "criticality_level":  {"dtype": "cat",      "required": True},
        "installation_date":  {"dtype": "datetime", "required": False},
        "warranty_expiry_date":{"dtype":"datetime", "required": False},
        "active_year":        {"dtype": "num",      "required": False},
    },

    "sla_rules": {
        "sla_rule_id":                    {"dtype": "str", "required": False},
        "service_tier":                   {"dtype": "cat", "required": True},
        "ticket_priority":                {"dtype": "cat", "required": True},
        "response_time_target_minutes":   {"dtype": "num", "required": True},
        "resolution_time_target_minutes": {"dtype": "num", "required": True},
    },

    "hardware_telemetry": {
        "asset_id":                   {"dtype": "str",      "required": True},
        "timestamp":                  {"dtype": "datetime", "required": True},
        "cpu_utilization_percent":    {"dtype": "num",      "required": True},
        "memory_utilization_percent": {"dtype": "num",      "required": True},
        "disk_utilization_percent":   {"dtype": "num",      "required": False},
        "temperature_celsius":        {"dtype": "num",      "required": False},
        "network_latency_ms":         {"dtype": "num",      "required": False},
        "packet_loss_percent":        {"dtype": "num",      "required": False},
        "error_count":                {"dtype": "num",      "required": False},
        "failure_flag":               {"dtype": "num",      "required": True},
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE POOLS
# ─────────────────────────────────────────────────────────────────────────────
# Columns explicitly excluded from model features regardless of dtype.
# ─────────────────────────────────────────────────────────────────────────────
ALWAYS_EXCLUDE = {
    # IDs and keys (high cardinality, no signal)
    "ticket_id", "year_ticket_id", "client_id", "asset_id",
    "engineer_id", "sla_rule_id", "active_year",
    # Timestamps (converted to derived features instead)
    "ticket_created_timestamp", "ticket_close_timestamp",
    "contract_start_date", "contract_end_date",
    "installation_date", "warranty_expiry_date",
    "telemetry_year", "timestamp",
    # Display fields
    "engineer_name", "client_name",
    # ticket_description excluded — NLP uses ticket_detailed_description only
    "ticket_description", "combined_text",
    # Post-resolution leakage fields
    "resolution_notes", "resolution_time_minutes",   # leakage for at-creation models
    "sla_breach_flag", "escalation_flag", "ticket_reopen_flag",
}

# Columns that are always treated as categorical regardless of inferred dtype
FORCE_CATEGORICAL = {
    "ticket_category", "issue_type", "ticket_priority", "ticket_status",
    "ticket_channel", "device_type", "manufacturer", "criticality_level",
    "industry_sector", "region", "service_tier", "specialization",
    "support_level", "shift_type",
}

# Columns that are always treated as numeric
FORCE_NUMERIC = {
    "experience_years", "first_response_time_minutes",
    "cpu_utilization_percent", "memory_utilization_percent",
    "disk_utilization_percent", "temperature_celsius",
    "network_latency_ms", "packet_loss_percent",
    "error_count", "failure_flag",
    "response_time_target_minutes", "resolution_time_target_minutes",
}


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class SchemaChangeError(Exception):
    """Raised when a required column is missing from a table."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_diff(df: pd.DataFrame, table_name: str,
                      year: Optional[int] = None,
                      log_path: Optional[Path] = None,
                      raise_on_required: bool = True) -> dict:
    """
    Compare a loaded DataFrame against the schema registry for table_name.
    
    Returns a diff dict with:
      added    → columns present in df but NOT in registry
      removed  → columns in registry but NOT in df
      required_missing → required columns that are absent (triggers SchemaChangeError)
      type_changed → columns where observed dtype differs from registered dtype
      
    All findings are printed and optionally appended to a log file.
    """
    table_key = table_name.lower()
    if table_key not in SCHEMA_REGISTRY:
        print(f"  ℹ️  '{table_name}' not in schema registry — treating all columns as candidate features.")
        return {"added": list(df.columns), "removed": [], "required_missing": [], "type_changed": []}

    expected = SCHEMA_REGISTRY[table_key]
    actual   = set(df.columns.str.lower())
    expected_cols = set(expected.keys())

    added   = sorted(actual - expected_cols)
    removed = sorted(expected_cols - actual)
    required_missing = [c for c in removed if expected[c]["required"]]
    optional_missing = [c for c in removed if not expected[c]["required"]]

    # Dtype drift detection
    type_changed = []
    for col in (actual & expected_cols):
        exp_dtype = expected[col]["dtype"]
        obs_dtype = str(df[col].dtype)
        mismatch = False
        if exp_dtype == "num"      and not pd.api.types.is_numeric_dtype(df[col]):     mismatch = True
        if exp_dtype == "datetime" and not pd.api.types.is_datetime64_any_dtype(df[col]): mismatch = True
        if mismatch:
            type_changed.append({"column": col, "expected": exp_dtype, "observed": obs_dtype})

    # ── Print report ──────────────────────────────────────────────────────
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    yr_lbl = f" [{year}]" if year else ""

    if not added and not removed and not type_changed:
        print(f"  ✅ {table_name}{yr_lbl}: schema matches registry ({len(df.columns)} columns)")
    else:
        print(f"\n  {'='*60}")
        print(f"  SCHEMA DIFF — {table_name}{yr_lbl}  at {ts}")
        print(f"  {'='*60}")
        if added:
            print(f"  ➕ NEW columns ({len(added)}) — will be auto-classified as candidate features:")
            for c in added:
                dtype_hint = _infer_dtype_hint(df[c])
                print(f"     + {c:40s} [{dtype_hint}]")
        if optional_missing:
            print(f"  ➖ REMOVED optional columns ({len(optional_missing)}) — models will run without them:")
            for c in optional_missing:
                print(f"     - {c}")
        if required_missing:
            print(f"  ❌ REMOVED REQUIRED columns ({len(required_missing)}) — pipeline cannot continue:")
            for c in required_missing:
                print(f"     ✗ {c}  (required={expected[c]['required']})")
        if type_changed:
            print(f"  ⚠️  TYPE CHANGES ({len(type_changed)}):")
            for tc in type_changed:
                print(f"     ~ {tc['column']:35s} expected={tc['expected']} got={tc['observed']}")
        print(f"  {'='*60}\n")

    # ── Optional log file ─────────────────────────────────────────────────
    if log_path:
        entry = {
            "timestamp": ts, "table": table_name, "year": year,
            "added": added, "removed_optional": optional_missing,
            "required_missing": required_missing, "type_changed": type_changed,
        }
        log_path = Path(log_path)
        existing = json.loads(log_path.read_text()) if log_path.exists() else []
        existing.append(entry)
        log_path.write_text(json.dumps(existing, indent=2, default=str))

    # ── Raise if required columns are missing ─────────────────────────────
    if required_missing and raise_on_required:
        raise SchemaChangeError(
            f"Required columns missing from {table_name}: {required_missing}. "
            f"Update the source data or mark these columns as optional in SCHEMA_REGISTRY."
        )

    return {
        "added": added,
        "removed_optional": optional_missing,
        "required_missing": required_missing,
        "type_changed": type_changed,
    }


def _infer_dtype_hint(series: pd.Series) -> str:
    """Classify a Series as num, cat, text, or datetime for auto-discovery."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "num"
    if series.dtype == object:
        avg_len = series.dropna().astype(str).str.len().mean()
        n_unique = series.nunique()
        n_total  = len(series)
        if avg_len > 30 or n_unique / max(n_total, 1) > 0.8:
            return "text"
        return "cat"
    return "unknown"


def dynamic_feature_pools(df: pd.DataFrame,
                          schema_diff: dict,
                          existing_cats: list,
                          existing_nums: list,
                          target_col: str = None,
                          verbose: bool = True) -> tuple[list, list, list]:
    """
    Build dynamic feature pools from a merged DataFrame.

    1. Start with the registered known features (existing_cats, existing_nums).
    2. Remove columns that are no longer present.
    3. Auto-classify NEW columns from schema_diff["added"] as candidate features.
    4. Return (cat_features, num_features, new_columns_flagged).

    Parameters
    ----------
    df              : the merged analytical base DataFrame
    schema_diff     : output from validate_and_diff()
    existing_cats   : the baseline categorical feature list
    existing_nums   : the baseline numeric feature list
    target_col      : exclude this column from features
    verbose         : print feature pool changes

    Returns
    -------
    cats      : final categorical feature list
    nums      : final numeric feature list
    new_flags : list of new columns auto-added (for review)
    """
    exclude = ALWAYS_EXCLUDE.copy()
    if target_col:
        exclude.add(target_col)

    # Start from registered lists, filtered to what actually exists in df
    cats = [c for c in existing_cats if c in df.columns and c not in exclude]
    nums = [c for c in existing_nums if c in df.columns and c not in exclude]

    # Track removals
    dropped_cats = [c for c in existing_cats if c not in df.columns]
    dropped_nums = [c for c in existing_nums if c not in df.columns]

    # Auto-classify new columns
    new_flags = []
    for col in schema_diff.get("added", []):
        if col in exclude or col in cats or col in nums:
            continue
        if col in FORCE_CATEGORICAL:
            cats.append(col)
            new_flags.append({"column": col, "classified_as": "categorical", "action": "auto-added"})
        elif col in FORCE_NUMERIC:
            nums.append(col)
            new_flags.append({"column": col, "classified_as": "numeric", "action": "auto-added"})
        else:
            hint = _infer_dtype_hint(df[col]) if col in df.columns else "unknown"
            if hint == "num":
                nums.append(col)
                new_flags.append({"column": col, "classified_as": "numeric (inferred)", "action": "auto-added — REVIEW"})
            elif hint == "cat":
                cats.append(col)
                new_flags.append({"column": col, "classified_as": "categorical (inferred)", "action": "auto-added — REVIEW"})
            else:
                new_flags.append({"column": col, "classified_as": hint, "action": "SKIPPED — text/datetime/unknown"})

    if verbose:
        print("\n  ── Feature pool update ──────────────────────────────────────")
        if dropped_cats or dropped_nums:
            print(f"  Removed (no longer in data):")
            for c in dropped_cats: print(f"    - {c} [categorical]")
            for c in dropped_nums: print(f"    - {c} [numeric]")
        if new_flags:
            print(f"  New columns auto-classified:")
            for nf in new_flags:
                flag = "⚠️ REVIEW" if "REVIEW" in nf["action"] else "✅"
                print(f"    {flag} {nf['column']:35s} → {nf['classified_as']}")
        print(f"  Final pools: {len(cats)} categorical, {len(nums)} numeric")
        print("  ─────────────────────────────────────────────────────────────\n")

    return cats, nums, new_flags


def validate_engineer_master(df: pd.DataFrame, year: int = None) -> dict:
    """
    Validate ENGINEER_MASTER specifically, since NB3 hard-references
    specialization, experience_years, shift_type, support_level.
    
    Returns mapping of expected_name -> actual_name (handles renames).
    """
    required = ["engineer_id", "specialization", "experience_years", "shift_type", "support_level"]
    actual_cols = list(df.columns.str.lower())

    col_map = {}
    missing = []
    for req in required:
        if req in actual_cols:
            col_map[req] = req
        else:
            # Fuzzy match attempt
            candidates = [c for c in actual_cols if req.split("_")[0] in c]
            if candidates:
                col_map[req] = candidates[0]
                print(f"  ⚠️  '{req}' not found in ENGINEER_MASTER — using '{candidates[0]}' as proxy. "
                      f"Update SCHEMA_REGISTRY if this is intentional.")
            else:
                missing.append(req)

    if missing:
        raise SchemaChangeError(
            f"Critical ENGINEER_MASTER columns missing: {missing}. "
            f"Engineer scoring cannot proceed. Update source data or SCHEMA_REGISTRY."
        )

    yr_lbl = f" [{year}]" if year else ""
    print(f"  ✅ ENGINEER_MASTER{yr_lbl} validated. Column mapping: {col_map}")
    return col_map


def print_schema_summary(diffs: dict):
    """Print a one-page summary of all schema diffs at the end of the load phase."""
    total_added   = sum(len(v.get("added",[])) for v in diffs.values())
    total_removed = sum(len(v.get("removed_optional",[])) for v in diffs.values())
    total_required= sum(len(v.get("required_missing",[])) for v in diffs.values())
    total_type    = sum(len(v.get("type_changed",[])) for v in diffs.values())
    any_change    = total_added + total_removed + total_required + total_type

    print("\n" + "═"*60)
    print("SCHEMA AUDIT SUMMARY")
    print("═"*60)
    if any_change == 0:
        print("  ✅  All tables match the registry. No schema changes detected.")
    else:
        print(f"  New columns added    : {total_added}")
        print(f"  Optional cols removed: {total_removed}")
        print(f"  Required cols missing: {total_required}  {'❌ PIPELINE STOPPED' if total_required else ''}")
        print(f"  Type changes         : {total_type}")
        if total_added:
            print("\n  Review auto-classified new features before trusting model outputs.")
    print("═"*60 + "\n")
