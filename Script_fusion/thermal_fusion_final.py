"""
Thermal Comfort — Sensor + Survey Fusion Script
================================================
Merges car sensor data (CSV), survey responses (Excel) and participant
profiles (Excel) into a single dataset ready for AI/ML training.

WHAT THIS SCRIPT DOES:
  Step 1 — Load car sensor CSV (55 columns, relative time in seconds)
  Step 2 — Load survey Excel (responses with absolute timestamps)
  Step 2b — Load profiles Excel and merge by participant_id
  Step 3 — For each survey row, calculate offset from ignition_time and
            find the closest sensor reading using time alignment
  Step 4 — Merge all sensor columns into the survey row
  Step 5 — Export the final fused dataset as CSV and Excel

OUTPUT COLUMNS:
  - All survey columns (participant_id, body_part, thermal_sensation, etc.)
  - All profile columns (age, gender, height_cm, weight_kg, etc.)
  - sensor_time_sec: the matched sensor time in seconds since ignition
  - sensor_time_offset_sec: difference between survey time and matched sensor
  - All 55 sensor columns prefixed with "sensor_"

FILES NEEDED (same folder as this script):
  - Car sensor CSV    → set CAR_CSV below
  - Survey Excel      → set SURVEY_XLSX below
  - Profiles Excel    → set PROFILES_XLSX below

HOW TO USE:
  1. Set CAR_CSV, SURVEY_XLSX and PROFILES_XLSX to the correct filenames
  2. Run the script
  3. Output files are saved in the same folder
"""

import pandas as pd
import numpy as np
import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# CAR_CSV       = r"C:\UNIGOU\Zimmermann\14042026\KONA\csv sync\kona-pokus1_1.csv"
# SURVEY_XLSX   = r"C:\UNIGOU\Zimmermann\14042026\KONA\csv sync\survey_export.xlsx"
# PROFILES_XLSX = r"C:\UNIGOU\Zimmermann\14042026\KONA\csv sync\profiles_export.xlsx"
# OUTPUT_DIR    = r"C:\UNIGOU\Zimmermann\14042026\KONA\csv sync"
from pathlib import Path

BASE_DIR = Path.cwd()

CAR_CSV       = BASE_DIR / "kona-pokus1_1.csv"
SURVEY_XLSX   = BASE_DIR / "survey_export.xlsx"
PROFILES_XLSX = BASE_DIR / "profiles_export.xlsx"
OUTPUT_DIR    = BASE_DIR

CAR_ENCODING = "cp1250"   # encoding of the car CSV (cp1250 for Czech files)

# Maximum allowed time gap (seconds) between a survey response and the
# closest sensor reading. Rows exceeding this are flagged with a warning.
MAX_TIME_GAP_SEC = 30

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD CAR SENSOR DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("THERMAL COMFORT — SENSOR + SURVEY FUSION")
print("=" * 65)

if not os.path.exists(CAR_CSV):
    print(f"\n[ERROR] Car CSV not found: {CAR_CSV}")
    sys.exit(1)

df_car = pd.read_csv(CAR_CSV, encoding=CAR_ENCODING, sep=None, engine="python")

# Rename first column to time_sec and drop the empty last column
df_car = df_car.rename(columns={df_car.columns[0]: "time_sec"})
df_car = df_car.dropna(subset=["time_sec"])
df_car["time_sec"] = pd.to_numeric(df_car["time_sec"], errors="coerce")
df_car = df_car.dropna(subset=["time_sec"])

# Drop unnamed/empty columns
df_car = df_car.loc[:, ~df_car.columns.str.startswith("Unnamed")]

# Prefix all sensor columns with "sensor_" for clarity in the final dataset
sensor_cols = [c for c in df_car.columns if c != "time_sec"]
df_car = df_car.rename(columns={c: f"sensor_{c}" for c in sensor_cols})
df_car = df_car.sort_values("time_sec").reset_index(drop=True)

print(f"\n[Step 1] Car sensor data loaded")
print(f"         File     : {os.path.basename(CAR_CSV)}")
print(f"         Rows     : {len(df_car)}")
print(f"         Sensors  : {len(sensor_cols)} columns")
print(f"         Duration : {df_car['time_sec'].min():.1f}s – {df_car['time_sec'].max():.1f}s "
      f"({df_car['time_sec'].max()/60:.1f} min)")
print(f"         Sampling : ~{df_car['time_sec'].diff().median():.3f}s interval")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LOAD SURVEY DATA
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(SURVEY_XLSX):
    print(f"\n[ERROR] Survey file not found: {SURVEY_XLSX}")
    sys.exit(1)

df_survey = pd.read_excel(SURVEY_XLSX)

# Parse timestamps — handle both UTC and naive datetimes
for col in ["timestamp", "ignition_time"]:
    if col in df_survey.columns:
        df_survey[col] = pd.to_datetime(df_survey[col], utc=True, errors="coerce")
        df_survey[col] = df_survey[col].dt.tz_localize(None)

# Drop rows with missing ignition_time (cannot align without it)
missing_ignition = df_survey["ignition_time"].isna().sum()
if missing_ignition > 0:
    print(f"\n[WARNING] {missing_ignition} rows have no ignition_time and will be skipped.")
    df_survey = df_survey.dropna(subset=["ignition_time"])

print(f"\n[Step 2] Survey data loaded")
print(f"         File         : {os.path.basename(SURVEY_XLSX)}")
print(f"         Total rows   : {len(df_survey)}")
print(f"         Participants : {df_survey['participant_id'].nunique()}")
print(f"         Body parts   : {df_survey['body_part'].nunique()}")
print(f"         Sessions     : {df_survey['ignition_time'].nunique()} unique ignition times")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2b — LOAD AND MERGE PARTICIPANT PROFILES
# Each survey row gets the profile columns of its participant added.
# Profile columns are prefixed with "profile_" for clarity.
# Rows with no matching profile are kept but profile columns will be NaN.
# ─────────────────────────────────────────────────────────────────────────────
if os.path.exists(PROFILES_XLSX):
    df_profiles = pd.read_excel(PROFILES_XLSX, sheet_name="Profiles")

    # Drop columns not useful for ML (email, created_at)
    drop_cols = [c for c in ["email", "created_at"] if c in df_profiles.columns]
    df_profiles = df_profiles.drop(columns=drop_cols)

    # Prefix profile columns (except participant_id which is the join key)
    profile_cols = [c for c in df_profiles.columns if c != "participant_id"]
    df_profiles  = df_profiles.rename(columns={c: f"profile_{c}" for c in profile_cols})

    # Merge into survey by participant_id (left join — keep all survey rows)
    df_survey = df_survey.merge(df_profiles, on="participant_id", how="left")

    matched   = df_survey["profile_age"].notna().sum() if "profile_age" in df_survey.columns else 0
    unmatched = len(df_survey) - matched

    print(f"\n[Step 2b] Participant profiles merged")
    print(f"          File            : {os.path.basename(PROFILES_XLSX)}")
    print(f"          Profile columns : {list(df_profiles.columns)}")
    print(f"          Rows matched    : {matched}")
    if unmatched > 0:
        print(f"          Rows unmatched  : {unmatched} (participant_id not found in profiles)")
else:
    print(f"\n[Step 2b] Profiles file not found ({os.path.basename(PROFILES_XLSX)}) — skipping.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TIME ALIGNMENT
# For each survey row:
#   offset_sec = (timestamp - ignition_time).total_seconds()
#   find the car sensor row where time_sec is closest to offset_sec
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Step 3] Aligning survey responses with sensor readings...")

def align_survey_to_sensors(df_survey_row, df_car):
    """
    Given a survey row, compute the time offset since ignition and find
    the closest sensor row. Returns the matched sensor row as a Series.
    """
    offset_sec = (df_survey_row["timestamp"] - df_survey_row["ignition_time"]).total_seconds()
    idx = (df_car["time_sec"] - offset_sec).abs().idxmin()
    matched = df_car.loc[idx].copy()
    matched["sensor_time_sec"]        = matched["time_sec"]
    matched["sensor_time_offset_sec"] = abs(df_car.loc[idx, "time_sec"] - offset_sec)
    matched["survey_offset_sec"]      = offset_sec
    return matched

# Apply alignment to every survey row
sensor_matches = df_survey.apply(
    lambda row: align_survey_to_sensors(row, df_car), axis=1
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — MERGE
# Concatenate survey columns + matched sensor columns side by side
# ─────────────────────────────────────────────────────────────────────────────
print(f"[Step 4] Merging survey and sensor data...")

# Drop time_sec from sensor matches (already stored as sensor_time_sec)
sensor_matches = sensor_matches.drop(columns=["time_sec"], errors="ignore")

df_fused = pd.concat(
    [df_survey.reset_index(drop=True), sensor_matches.reset_index(drop=True)],
    axis=1
)

# Flag rows where the time gap is large
large_gap = df_fused["sensor_time_offset_sec"] > MAX_TIME_GAP_SEC
if large_gap.sum() > 0:
    print(f"\n[WARNING] {large_gap.sum()} rows have a sensor time gap > {MAX_TIME_GAP_SEC}s.")
    print(f"          These may indicate misaligned ignition_time.")
    print(f"          They are kept in the dataset but flagged in column 'time_gap_warning'.")
    df_fused["time_gap_warning"] = large_gap
else:
    df_fused["time_gap_warning"] = False

print(f"\n         Survey rows    : {len(df_survey)}")
print(f"         Fused columns  : {len(df_fused.columns)}")
print(f"           → Survey cols  : {len(df_survey.columns)}")
print(f"           → Sensor cols  : {len(sensor_cols)}")
print(f"           → Meta cols    : 3 (sensor_time_sec, sensor_time_offset_sec, survey_offset_sec)")
print(f"         Large gap rows  : {large_gap.sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4b — PIVOT: one row per timestamp
# Transform 14 rows per timestamp (one per body part) into 1 row per timestamp
# with 42 columns (14 body parts × 3 metrics: thermal_sensation, thermal_comfort, wanted_action)
#
# Body part name → column prefix mapping (spaces/special chars replaced with _)
# Example: "Upper back" → "upper_back_thermal_comfort"
#          "Whole body (initial)" → "whole_body_initial_thermal_sensation"
#          "Whole body (overall)" → "whole_body_overall_thermal_comfort"
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Step 4b] Pivoting to one row per timestamp...")

def body_part_to_key(bp):
    """Convert body part label to a clean column prefix."""
    return (bp.lower()
              .replace("whole body (initial)", "whole_body_initial")
              .replace("whole body (overall)", "whole_body_overall")
              .replace(" + ", "_")
              .replace(" ", "_")
              .replace("(", "")
              .replace(")", ""))

# Metadata columns that are the same for all body parts in a timestamp
# (participant info, sensors, context) — these stay as single columns
meta_cols = [c for c in df_fused.columns
             if c not in ["body_part", "thermal_sensation", "thermal_comfort", "wanted_action"]]

# Pivot the 3 response metrics by body_part
df_pivot = df_fused.pivot_table(
    index="timestamp",
    columns="body_part",
    values=["thermal_sensation", "thermal_comfort", "wanted_action"],
    aggfunc="first"
)

# Flatten multi-level columns: (metric, body_part) → "body_part_key_metric"
df_pivot.columns = [
    f"{body_part_to_key(bp)}_{metric}"
    for metric, bp in df_pivot.columns
]
df_pivot = df_pivot.reset_index()

# Get one row of metadata per timestamp (drop duplicate timestamps)
df_meta = df_fused[meta_cols].drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

# Merge metadata + pivoted responses
df_wide = df_meta.merge(df_pivot, on="timestamp", how="left")

print(f"         Rows before pivot : {len(df_fused)}")
print(f"         Rows after pivot  : {len(df_wide)}")
print(f"         Response columns  : {[c for c in df_wide.columns if any(m in c for m in ['_thermal_', '_wanted_'])][:6]} ...")
print(f"         Total columns     : {len(df_wide.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — EXPORT
# Wide format only, with timestamp in filename to avoid overwriting
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Step 5] Exporting dataset...")

from datetime import datetime
run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

out_wide_csv   = os.path.join(OUTPUT_DIR, f"thermal_comfort_fused_{run_ts}.csv")
out_wide_excel = os.path.join(OUTPUT_DIR, f"thermal_comfort_fused_{run_ts}.xlsx")

df_wide.to_csv(out_wide_csv,   index=False, encoding="utf-8-sig")
df_wide.to_excel(out_wide_excel, index=False)

print(f"         Saved: {os.path.basename(out_wide_csv)}")
print(f"         Saved: {os.path.basename(out_wide_excel)}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"FUSION COMPLETE")
print(f"{'=' * 65}")
print(f"  Wide format : {len(df_wide)} rows × {len(df_wide.columns)} cols")
print(f"  Saved as    : {os.path.basename(out_wide_csv)}")
print(f"\n  Response columns sample:")
sample_response_cols = [c for c in df_wide.columns if any(m in c for m in ['_thermal_', '_wanted_'])][:8]
print(f"    {sample_response_cols}")
print(f"\n  Sample data (first 3 rows, key columns):")
sample_cols = ["timestamp", "participant_id"] + sample_response_cols[:4]
sample_cols = [c for c in sample_cols if c in df_wide.columns]
print(df_wide[sample_cols].head(3).to_string(index=False))
