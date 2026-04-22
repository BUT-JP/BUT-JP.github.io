"""
Thermal Comfort — Sensor + Survey Fusion Script
================================================
Merges car sensor data (CSV), survey responses (Excel) and participant
profiles (Excel) into a single dataset ready for AI/ML training.

WHAT THIS SCRIPT DOES:
  Step 1 — Load one or more car sensor CSVs
  Step 2 — Load one or more survey Excels and combine them
  Step 2b — Load one or more profiles Excels and merge by participant_id
  Step 3 — For each survey row, align to the correct car CSV using ignition_time
  Step 4 — Merge all sensor columns into the survey row
  Step 4b — Pivot to one row per timestamp (wide format)
  Step 5 — Export the final fused dataset as CSV and Excel

HOW TO USE:
  1. Add your car CSV files to CAR_CSV_FILES below
  2. Add your survey Excel exports to SURVEY_XLSX_FILES below
  3. Add your profiles Excel exports to PROFILES_XLSX_FILES below
  4. Run the script
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

# Add as many car CSV files as needed — one per session
CAR_CSV_FILES = [
    r"C:\UNIGOU\Zimmermann\14042026\KONA\csv sync\kona-pokus1_1.csv",
    # r"C:\UNIGOU\Zimmermann\14042026\KONA\csv sync\kona-sessao2.csv",
]

# Add all survey Excel exports — duplicates are removed automatically
SURVEY_XLSX_FILES = [
    r"C:\UNIGOU\Zimmermann\14042026\KONA\csv sync\survey_export.xlsx",
    # r"C:\UNIGOU\Zimmermann\14042026\KONA\csv sync\survey_export_april.xlsx",
]

# Add all profiles Excel exports — duplicates are removed automatically
PROFILES_XLSX_FILES = [
    r"C:\UNIGOU\Zimmermann\14042026\KONA\csv sync\profiles_export.xlsx",
    # r"C:\UNIGOU\Zimmermann\14042026\KONA\csv sync\profiles_export_april.xlsx",
]

OUTPUT_DIR       = r"C:\UNIGOU\Zimmermann\14042026\KONA\csv sync"
CAR_ENCODING     = "cp1250"
MAX_TIME_GAP_SEC = 30

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_car_csv(path, encoding):
    """Load a car sensor CSV, rename time column, prefix sensor columns."""
    df = pd.read_csv(path, encoding=encoding, sep=None, engine="python")
    df = df.rename(columns={df.columns[0]: "time_sec"})
    df = df.dropna(subset=["time_sec"])
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    df = df.dropna(subset=["time_sec"])
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    sensor_cols = [c for c in df.columns if c != "time_sec"]
    df = df.rename(columns={c: f"sensor_{c}" for c in sensor_cols})
    df = df.sort_values("time_sec").reset_index(drop=True)
    return df, sensor_cols

def align_survey_to_sensors(survey_row, df_car):
    """Find the closest sensor row for a given survey response."""
    offset_sec = (survey_row["timestamp"] - survey_row["ignition_time"]).total_seconds()
    idx = (df_car["time_sec"] - offset_sec).abs().idxmin()
    matched = df_car.loc[idx].copy()
    matched["sensor_time_sec"]        = matched["time_sec"]
    matched["sensor_time_offset_sec"] = abs(df_car.loc[idx, "time_sec"] - offset_sec)
    matched["survey_offset_sec"]      = offset_sec
    matched["car_csv_file"]           = df_car.attrs.get("source_file", "unknown")
    return matched

def body_part_to_key(bp):
    """Convert body part label to a clean column prefix."""
    return (bp.lower()
              .replace("whole body (initial)", "whole_body_initial")
              .replace("whole body (overall)", "whole_body_overall")
              .replace(" + ", "_")
              .replace(" ", "_")
              .replace("(", "")
              .replace(")", ""))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD ALL CAR CSVs
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("THERMAL COMFORT — SENSOR + SURVEY FUSION")
print("=" * 65)

car_datasets = {}   # dict: filename → df_car
all_sensor_cols = None

for car_path in CAR_CSV_FILES:
    if not os.path.exists(car_path):
        print(f"\n[ERROR] Car CSV not found: {car_path}")
        sys.exit(1)

    df_car, sensor_cols = load_car_csv(car_path, CAR_ENCODING)
    df_car.attrs["source_file"] = os.path.basename(car_path)
    car_datasets[car_path] = df_car

    # Use sensor columns from first file (assumed consistent across sessions)
    if all_sensor_cols is None:
        all_sensor_cols = sensor_cols

    print(f"\n[Step 1] Car CSV loaded: {os.path.basename(car_path)}")
    print(f"         Rows     : {len(df_car)}")
    print(f"         Sensors  : {len(sensor_cols)} columns")
    print(f"         Duration : {df_car['time_sec'].min():.1f}s – {df_car['time_sec'].max():.1f}s "
          f"({df_car['time_sec'].max()/60:.1f} min)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LOAD SURVEY DATA (one or more files, duplicates removed)
# ─────────────────────────────────────────────────────────────────────────────
survey_frames = []
for survey_path in SURVEY_XLSX_FILES:
    if not os.path.exists(survey_path):
        print(f"\n[WARNING] Survey file not found, skipping: {survey_path}")
        continue
    df_s = pd.read_excel(survey_path)
    df_s["_source_file"] = os.path.basename(survey_path)
    survey_frames.append(df_s)
    print(f"\n[Step 2] Survey loaded: {os.path.basename(survey_path)} ({len(df_s)} rows)")

if not survey_frames:
    print("\n[ERROR] No survey files found.")
    sys.exit(1)

df_survey = pd.concat(survey_frames, ignore_index=True)

# Remove duplicate rows (same timestamp + participant + body_part)
before_dedup = len(df_survey)
df_survey = df_survey.drop_duplicates(subset=["timestamp", "participant_id", "body_part"])
after_dedup = len(df_survey)
if before_dedup > after_dedup:
    print(f"\n         Removed {before_dedup - after_dedup} duplicate rows across files")

for col in ["timestamp", "ignition_time"]:
    if col in df_survey.columns:
        df_survey[col] = pd.to_datetime(df_survey[col], utc=True, errors="coerce")
        df_survey[col] = df_survey[col].dt.tz_localize(None)

missing_ignition = df_survey["ignition_time"].isna().sum()
if missing_ignition > 0:
    print(f"\n[WARNING] {missing_ignition} rows have no ignition_time and will be skipped.")
    df_survey = df_survey.dropna(subset=["ignition_time"])

print(f"\n         Total rows after merge : {len(df_survey)}")
print(f"         Participants           : {df_survey['participant_id'].nunique()}")
print(f"         Sessions               : {df_survey['ignition_time'].nunique()} unique ignition times")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2b — LOAD AND MERGE PARTICIPANT PROFILES (one or more files)
# ─────────────────────────────────────────────────────────────────────────────
profile_frames = []
for prof_path in PROFILES_XLSX_FILES:
    if not os.path.exists(prof_path):
        print(f"\n[WARNING] Profiles file not found, skipping: {prof_path}")
        continue
    try:
        df_p = pd.read_excel(prof_path, sheet_name="Profiles")
        profile_frames.append(df_p)
        print(f"\n[Step 2b] Profiles loaded: {os.path.basename(prof_path)} ({len(df_p)} rows)")
    except Exception as e:
        print(f"\n[WARNING] Could not read Profiles sheet from {prof_path}: {e}")

if profile_frames:
    df_profiles  = pd.concat(profile_frames, ignore_index=True)
    df_profiles  = df_profiles.drop_duplicates(subset=["participant_id"])
    drop_cols    = [c for c in ["email", "created_at"] if c in df_profiles.columns]
    df_profiles  = df_profiles.drop(columns=drop_cols)
    profile_cols = [c for c in df_profiles.columns if c != "participant_id"]
    df_profiles  = df_profiles.rename(columns={c: f"profile_{c}" for c in profile_cols})
    df_survey    = df_survey.merge(df_profiles, on="participant_id", how="left")
    matched      = df_survey["profile_age"].notna().sum() if "profile_age" in df_survey.columns else 0
    print(f"         Total profiles: {len(df_profiles)} | Rows matched: {matched}")
else:
    print(f"\n[Step 2b] No profiles files found — skipping.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TIME ALIGNMENT
# Each survey row is matched to the correct car CSV using ignition_time.
# If multiple CSVs exist, the one whose session time range covers the
# survey offset is used. If only one CSV exists, it is used for all rows.
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Step 3] Aligning survey responses with sensor readings...")

def find_best_car_csv(survey_row, car_datasets):
    """
    Find the most appropriate car CSV for a survey row.
    If multiple CSVs exist, pick the one whose time range best covers
    the survey offset. Falls back to the first CSV if none match.
    """
    offset_sec = (survey_row["timestamp"] - survey_row["ignition_time"]).total_seconds()

    if len(car_datasets) == 1:
        return list(car_datasets.values())[0]

    # Try to find a CSV whose time range covers the offset
    for df_car in car_datasets.values():
        t_min = df_car["time_sec"].min()
        t_max = df_car["time_sec"].max()
        if t_min <= offset_sec <= t_max:
            return df_car

    # Fallback: use the CSV with the closest max time
    best = min(car_datasets.values(),
               key=lambda d: abs(d["time_sec"].max() - offset_sec))
    return best

sensor_matches = df_survey.apply(
    lambda row: align_survey_to_sensors(row, find_best_car_csv(row, car_datasets)),
    axis=1
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — MERGE
# ─────────────────────────────────────────────────────────────────────────────
print(f"[Step 4] Merging survey and sensor data...")

sensor_matches = sensor_matches.drop(columns=["time_sec"], errors="ignore")

df_fused = pd.concat(
    [df_survey.reset_index(drop=True), sensor_matches.reset_index(drop=True)],
    axis=1
)

large_gap = df_fused["sensor_time_offset_sec"] > MAX_TIME_GAP_SEC
df_fused["time_gap_warning"] = large_gap

if large_gap.sum() > 0:
    print(f"\n[WARNING] {large_gap.sum()} rows have sensor gap > {MAX_TIME_GAP_SEC}s.")

print(f"\n         Survey rows   : {len(df_survey)}")
print(f"         Fused columns : {len(df_fused.columns)}")
print(f"         Large gaps    : {large_gap.sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4b — PIVOT: one row per timestamp
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Step 4b] Pivoting to one row per timestamp...")

meta_cols = [c for c in df_fused.columns
             if c not in ["body_part", "thermal_sensation", "thermal_comfort", "wanted_action"]]

df_pivot = df_fused.pivot_table(
    index="timestamp",
    columns="body_part",
    values=["thermal_sensation", "thermal_comfort", "wanted_action"],
    aggfunc="first"
)

df_pivot.columns = [
    f"{body_part_to_key(bp)}_{metric}"
    for metric, bp in df_pivot.columns
]
df_pivot = df_pivot.reset_index()

df_meta = df_fused[meta_cols].drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
df_wide = df_meta.merge(df_pivot, on="timestamp", how="left")

print(f"         Rows before pivot : {len(df_fused)}")
print(f"         Rows after pivot  : {len(df_wide)}")
print(f"         Total columns     : {len(df_wide.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — EXPORT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[Step 5] Exporting dataset...")

run_ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
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
print(f"  Car CSVs processed : {len(CAR_CSV_FILES)}")
print(f"  Wide format        : {len(df_wide)} rows × {len(df_wide.columns)} cols")
print(f"  Saved as           : {os.path.basename(out_wide_csv)}")
sample_response_cols = [c for c in df_wide.columns if any(m in c for m in ['_thermal_', '_wanted_'])][:6]
print(f"\n  Sample response columns: {sample_response_cols}")
sample_cols = ["timestamp", "participant_id", "car_csv_file"] + sample_response_cols[:3]
sample_cols = [c for c in sample_cols if c in df_wide.columns]
print(f"\n  Sample data (first 3 rows):")
print(df_wide[sample_cols].head(3).to_string(index=False))
