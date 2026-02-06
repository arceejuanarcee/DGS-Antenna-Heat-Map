#!/usr/bin/env python3
"""
Combined GUI: 
  1) DGS Fault & Track Mapper (CSV/XLSX/TXT) + Hover Tooltips
  2) Antenna Fault Frequency Heat Map (Az/El) — Error Code Only + Click Inspect

Goal: merge two existing tools into ONE GUI without changing their internal logic/behavior.
Implementation approach:
- Keep each tool's logic/functions intact.
- Convert each original Tk() app into a Frame-based "panel" so both can live in a single Tk root window.
- Use ttk.Notebook tabs to switch between tools.

Notes:
- You still need the same dependencies as before (pandas/numpy/matplotlib; mplcursors optional for hover).
"""

from __future__ import annotations

# ---- shared stdlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict

# ---- third-party
import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Patch, Rectangle
from matplotlib import cm

# --- Optional hover tooltips (for DGS Fault & Track Mapper)
try:
    import mplcursors
    MPLCURSORS_AVAILABLE = True
except Exception:
    MPLCURSORS_AVAILABLE = False


# =============================================================================
# TAB 1: DGS Fault & Track Mapper (logic kept the same; only GUI host changed)
# =============================================================================

# ============================== Config ======================================

FAULT_LIKE = {"fault", "error", "alarm"}
TRACK_LIKE = {"track", "pass", "session", "tracking"}

COL_DATETIME = ["timestamp", "datetime", "dt", "time", "time_utc", "time_local"]
COL_DATE = ["date", "day"]
COL_TIME = ["time", "clock"]
COL_TYPE = ["type", "event", "category", "kind"]
COL_CODE = ["code", "error_code", "fault_code", "errcode", "err_code"]
COL_START = ["start", "start_time", "begin", "window_start"]
COL_END = ["end", "end_time", "finish", "window_end"]
COL_SAT = ["satellite", "sat", "spacecraft", "sc", "target_name"]

# max duration we accept for a track window during parsing/pairing (guards parsing mistakes)
MAX_TRACK_MINUTES = 120  # 2 hours
# if a start has no stop, we synthesize a short window
SYNTH_WINDOW_MINUTES = 10
# when pairing start→nearest stop, maximum gap to accept
PAIR_MAX_GAP_MINUTES = 120

# --- Plot filters (requested)
MAX_PLOT_TRACK_MINUTES = 20          # disregard windows longer than this (NOT PLOTTED)
FULL_DAY_EPS_MINUTES = 1             # treat within 1 minute of 24h as full-day-ish


# ============================== Regexes =====================================

LOG_TS = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*,\s*(.*)$"
)
LOG_TRACK_RANGE = re.compile(
    r"\((\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?),\s*"
    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\)"
)
LOG_ERROR_CODE = re.compile(r"(?:Error\s*code|Err\.?\s*code)\s*(\d+)", flags=re.I)
LOG_SAT_NAME = re.compile(
    r"Track:\s*(?:Launching|Intercepting)?\s*([A-Z0-9][A-Z0-9\s\-]+?)\s*\(",
    flags=0,
)


# ============================== Helpers =====================================

def first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in cols:
            return cols[name]
    return None


_TIME_24_RE = re.compile(r"(\b24:00(?::00(?:\.0+)?)?\b)")
_EXPLICIT_DT_FORMATS = [
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S.%f",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S.%f",
    "%d/%m/%Y %H:%M:%S",
    "%m-%d-%y %H:%M:%S",
    "%d-%m-%y %H:%M:%S",
    "%m-%d-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
    "%H:%M:%S.%f",
    "%H:%M:%S",
]


def _strip_tz(x: pd.Series) -> pd.Series:
    try:
        return x.dt.tz_convert(None)
    except Exception:
        try:
            return x.dt.tz_localize(None)
        except Exception:
            return x


def _fix_24h(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        return s.str.replace(_TIME_24_RE, "23:59:59.999", regex=True)
    return s


def parse_dt(series: pd.Series) -> pd.Series:
    raw = series.astype(str)
    raw = _fix_24h(raw)
    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    mask = out.isna()
    for fmt in _EXPLICIT_DT_FORMATS:
        try:
            trial = pd.to_datetime(raw[mask], format=fmt, errors="coerce")
            out.loc[mask] = trial
            mask = out.isna()
            if not mask.any():
                break
        except Exception:
            pass

    if out.isna().any():
        fallback = pd.to_datetime(raw[out.isna()], errors="coerce")
        out.loc[out.isna()] = fallback

    return _strip_tz(out)


def parse_date_time(date_s: pd.Series, time_s: pd.Series) -> pd.Series:
    combined = date_s.astype(str).str.strip() + " " + _fix_24h(time_s.astype(str)).str.strip()
    return parse_dt(combined)


# ============================== TXT Parser ==================================

def parse_txt_log(path: str) -> pd.DataFrame:
    """
    Produces rows with columns:
      timestamp (string),
      type ∈ {'fault','track'},
      code (string or None),
      start (string or None),
      end   (string or None),
      satellite (string or None)
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOG_TS.match(line)
            if not m:
                continue
            ts_str, rest = m.groups()
            rest_low = rest.lower()

            type_guess = None
            code_val = None
            start_dt = None
            end_dt = None
            sat_name = None

            # faults
            err_m = LOG_ERROR_CODE.search(rest)
            if "fault:" in rest_low and err_m:
                type_guess = "fault"
                code_val = err_m.group(1)

            # explicit windows
            tr_m = LOG_TRACK_RANGE.search(rest)
            if ("track:" in rest_low) and tr_m:
                type_guess = "track"
                start_dt, end_dt = tr_m.groups()
                sat_m = LOG_SAT_NAME.search(rest)
                if sat_m:
                    sat_name = sat_m.group(1).strip()

            # starts/stops markers
            if "event (track_start)" in rest_low:
                type_guess = "track"
                start_dt = ts_str
                end_dt = None
            if "event (track_stop)" in rest_low:
                type_guess = "track"
                start_dt = None
                end_dt = ts_str

            if type_guess is None:
                continue

            rows.append({
                "timestamp": ts_str,
                "type": type_guess,
                "code": code_val,
                "start": start_dt,
                "end": end_dt,
                "satellite": sat_name
            })
    return pd.DataFrame(rows)


def load_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx", ".xlsm"):
        return pd.read_excel(path)
    elif ext in (".txt", ".log"):
        return parse_txt_log(path)
    else:
        return pd.read_csv(path)


# ============================== Normalization ===============================

def _pair_starts_stops(track_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Build clean windows from:
      A) explicit rows with both dt_start & dt_end
      B) separate start-only and stop-only rows (pair within PAIR_MAX_GAP_MINUTES on same day)
      C) synthesize short window for unpaired starts
    """
    explicit = track_rows[track_rows["dt_start"].notna() & track_rows["dt_end"].notna()][
        ["dt_start", "dt_end", "satellite"]
    ].copy()

    starts = track_rows[track_rows["dt_start"].notna() & track_rows["dt_end"].isna()][
        ["dt_start", "satellite"]
    ].sort_values("dt_start").reset_index(drop=True)

    stops = track_rows[track_rows["dt_end"].notna() & track_rows["dt_start"].isna()][
        ["dt_end", "satellite"]
    ].sort_values("dt_end").reset_index(drop=True)

    paired = []
    j = 0
    for _, srow in starts.iterrows():
        s_dt = srow["dt_start"]
        s_day = s_dt.date()
        while j < len(stops) and stops.loc[j, "dt_end"] < s_dt:
            j += 1
        if j < len(stops):
            e_dt = stops.loc[j, "dt_end"]
            e_day = e_dt.date()
            gap = (e_dt - s_dt).total_seconds() / 60.0
            if s_day == e_day and 0 < gap <= PAIR_MAX_GAP_MINUTES:
                paired.append({"dt_start": s_dt, "dt_end": e_dt, "satellite": srow["satellite"]})
                j += 1
                continue
        paired.append({
            "dt_start": s_dt,
            "dt_end": s_dt + pd.Timedelta(minutes=SYNTH_WINDOW_MINUTES),
            "satellite": srow["satellite"]
        })

    out = pd.concat([explicit, pd.DataFrame(paired)], ignore_index=True) if len(paired) > 0 else explicit

    # sanitize duration + bounds
    if not out.empty:
        good = (out["dt_end"] > out["dt_start"]) & \
               ((out["dt_end"] - out["dt_start"]) <= pd.Timedelta(minutes=MAX_TRACK_MINUTES))
        out = out[good].copy()

    return out


def extract_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df2 = df.copy()
    df2.rename(columns={c: c.strip() for c in df2.columns}, inplace=True)

    type_col = first_col(df2, COL_TYPE)
    dt_col = first_col(df2, COL_DATETIME)
    date_col = first_col(df2, COL_DATE)
    time_col = first_col(df2, COL_TIME)
    code_col = first_col(df2, COL_CODE)
    start_col = first_col(df2, COL_START)
    end_col = first_col(df2, COL_END)
    sat_col = first_col(df2, COL_SAT)

    # base datetimes
    if dt_col is not None:
        dt_series = parse_dt(df2[dt_col])
    elif date_col is not None and time_col is not None:
        dt_series = parse_date_time(df2[date_col], df2[time_col])
    elif "timestamp" in df2.columns:
        dt_series = parse_dt(df2["timestamp"])
    else:
        dt_series = pd.Series(pd.NaT, index=df2.index, dtype="datetime64[ns]")

    dt_start = parse_dt(df2[start_col]) if start_col else parse_dt(df2["start"]) if "start" in df2.columns else pd.Series(pd.NaT, index=df2.index, dtype="datetime64[ns]")
    dt_end   = parse_dt(df2[end_col])   if end_col   else parse_dt(df2["end"])   if "end"   in df2.columns else pd.Series(pd.NaT, index=df2.index, dtype="datetime64[ns]")

    base_dt_mask = dt_series.isna() & dt_start.notna()
    dt_series.loc[base_dt_mask] = dt_start.loc[base_dt_mask]

    # code
    code_series = None
    if code_col is not None:
        code_series = pd.to_numeric(df2[code_col].astype(str).str.extract(r"(-?\d+)", expand=False), errors="coerce")
    elif "code" in df2.columns:
        code_series = pd.to_numeric(df2["code"], errors="coerce")

    # type
    if type_col is not None:
        type_vals = df2[type_col].astype(str).str.lower()
    elif "type" in df2.columns:
        type_vals = df2["type"].astype(str).str.lower()
    else:
        type_vals = pd.Series("", index=df2.index)

    # satellite
    if sat_col is not None:
        sat_vals = df2[sat_col].astype(str).str.strip()
    elif "satellite" in df2.columns:
        sat_vals = df2["satellite"].astype(str).str.strip()
    else:
        sat_vals = pd.Series(pd.NA, index=df2.index)

    base = pd.DataFrame({
        "dt": dt_series,
        "dt_start": dt_start,
        "dt_end": dt_end,
        "etype": type_vals,
        "satellite": sat_vals
    })

    # infer missing type
    if (base["etype"] == "").any():
        code_present = code_series.notna() if code_series is not None else pd.Series(False, index=base.index)
        inferred = np.where(code_present, "fault",
                            np.where(base["dt_start"].notna() | base["dt_end"].notna(), "track", "unknown"))
        base.loc[base["etype"] == "", "etype"] = inferred[base["etype"] == ""]

    # labels/positions
    base["day_label"] = base["dt"].dt.strftime("%m-%d-%Y")
    base.loc[base["dt"].isna() & base["dt_start"].notna(), "day_label"] = base["dt_start"].dt.strftime("%m-%d-%Y")
    # For correct chronological sorting on the x-axis
    base["day_dt"] = base["dt"].dt.date
    base.loc[base["dt"].isna() & base["dt_start"].notna(), "day_dt"] = base["dt_start"].dt.date
    base["hour_float"] = base["dt"].dt.hour.add(base["dt"].dt.minute.div(60)).astype(float)

    # ---- FAULTS
    faults = base[base["etype"].isin(FAULT_LIKE | {"fault"})].copy()
    if code_series is not None:
        faults["code_num"] = code_series
        faults = faults[faults["code_num"].notna()]
        faults["code_str"] = "Fault " + faults["code_num"].astype(int).astype(str)
    else:
        faults = faults.iloc[0:0].copy()
    faults = faults[faults["day_label"].notna() & faults["hour_float"].notna()]

    # ---- TRACKS (pair starts/stops + sanitize)
    tr_raw = base[base["etype"].isin(TRACK_LIKE | {"track"})][["dt", "dt_start", "dt_end", "satellite"]].copy()
    tr_raw["dt_start"] = tr_raw["dt_start"].where(tr_raw["dt_start"].notna(), tr_raw["dt"])
    tracks = _pair_starts_stops(tr_raw)

    if tracks.empty:
        return faults, tracks

    tracks["day_label"] = tracks["dt_start"].dt.strftime("%m-%d-%Y")
    tracks["day_dt"] = tracks["dt_start"].dt.date
    y1 = tracks["dt_start"].dt.hour + tracks["dt_start"].dt.minute.div(60)
    y2 = tracks["dt_end"].dt.hour + tracks["dt_end"].dt.minute.div(60)

    swap = y2 < y1
    y1, y2 = y1.where(~swap, y2), y2.where(~swap, y1)

    y1 = y1.clip(lower=0, upper=23.999)
    y2 = y2.clip(lower=0.001, upper=24.0)

    # remove full-day-ish windows early
    keep = tracks["day_label"].notna() & (y2 > y1) & ~((y1 <= 0.001) & (y2 >= 23.999))
    tracks = tracks[keep].copy()
    tracks["y1"] = y1.loc[tracks.index]
    tracks["y2"] = y2.loc[tracks.index]

    return faults, tracks


# ============================== Plotting ====================================

def _color_map(keys: List[str]) -> Dict[str, tuple]:
    cmap = cm.get_cmap('tab20')
    return {k: cmap(i % 20) for i, k in enumerate(keys)}


def _merge_overlaps_per_sat_day(tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Merge overlapping/adjacent windows for same (day_label, satellite).
    This reduces clutter and prevents stacked artifacts.
    """
    if tracks.empty:
        return tracks

    t = tracks.copy()
    sat_series = t["satellite"].astype("string")
    missing = sat_series.isna() | sat_series.str.strip().eq("") | sat_series.str.strip().str.lower().isin({"none", "nan"})
    t["satellite"] = sat_series.mask(missing, np.nan)

    out_rows = []
    for (day, sat), grp in t.groupby(["day_label", "satellite"], dropna=False):
        g = grp.sort_values(["y1", "y2"])
        cur_y1 = None
        cur_y2 = None
        cur_idx = None

        for _, r in g.iterrows():
            y1 = float(r["y1"])
            y2 = float(r["y2"])
            if cur_y1 is None:
                cur_y1, cur_y2 = y1, y2
                cur_idx = r
                continue

            # overlap or touching (within 1 minute)
            if y1 <= cur_y2 + (1/60):
                cur_y2 = max(cur_y2, y2)
            else:
                row = cur_idx.copy()
                row["y1"] = cur_y1
                row["y2"] = cur_y2
                out_rows.append(row)
                cur_y1, cur_y2 = y1, y2
                cur_idx = r

        if cur_y1 is not None:
            row = cur_idx.copy()
            row["y1"] = cur_y1
            row["y2"] = cur_y2
            out_rows.append(row)

    merged = pd.DataFrame(out_rows)
    merged = merged.drop_duplicates(subset=["day_label", "satellite", "y1", "y2"])
    return merged


def _filter_tracks_for_plot(tracks: pd.DataFrame) -> pd.DataFrame:
    """
    DISREGARD & DO NOT PLOT:
      - full-day-ish (≈24h) windows
      - > MAX_PLOT_TRACK_MINUTES windows
    """
    if tracks.empty:
        return tracks

    dur_min = (tracks["y2"].astype(float) - tracks["y1"].astype(float)) * 60.0
    full_day_min = 24.0 * 60.0

    is_full_dayish = dur_min >= (full_day_min - FULL_DAY_EPS_MINUTES)
    too_long = dur_min > MAX_PLOT_TRACK_MINUTES

    keep = (~is_full_dayish) & (~too_long) & (dur_min > 0)
    return tracks.loc[keep].copy()


def plot_fault_map(
    faults_list: List[pd.DataFrame],
    tracks_list: List[pd.DataFrame],
    month_title: Optional[str] = None
):
    faults = pd.concat(faults_list, ignore_index=True) if faults_list else pd.DataFrame(
        columns=["day_label", "hour_float", "code_str", "code_num", "dt"]
    )
    tracks = pd.concat(tracks_list, ignore_index=True) if tracks_list else pd.DataFrame(
        columns=["day_label", "y1", "y2", "satellite", "dt_start", "dt_end"]
    )

    # merge, then filter out long/full-day windows (so “tall borders” vanish)
    if not tracks.empty:
        tracks = _merge_overlaps_per_sat_day(tracks)
        tracks = _filter_tracks_for_plot(tracks)

    # Build x-axis days in true chronological order (not string-sorted)
    day_pairs = []
    if "day_dt" in faults.columns and "day_label" in faults.columns:
        day_pairs += list(zip(faults["day_dt"], faults["day_label"]))
    if "day_dt" in tracks.columns and "day_label" in tracks.columns:
        day_pairs += list(zip(tracks["day_dt"], tracks["day_label"]))

    # Fallback (should be rare): parse labels if day_dt missing
    if not day_pairs:
        for lab in pd.Index(faults.get("day_label", pd.Series(dtype=str))).dropna().unique().tolist():
            try:
                day_pairs.append((datetime.strptime(lab, "%m-%d-%Y").date(), lab))
            except Exception:
                pass
        for lab in pd.Index(tracks.get("day_label", pd.Series(dtype=str))).dropna().unique().tolist():
            try:
                day_pairs.append((datetime.strptime(lab, "%m-%d-%Y").date(), lab))
            except Exception:
                pass

    # Deduplicate by label, keep earliest date if duplicates occur
    best = {}
    for d, lab in day_pairs:
        if pd.isna(lab) or d is None:
            continue
        if lab not in best or d < best[lab]:
            best[lab] = d

    unique_days = [lab for (lab, _) in sorted(best.items(), key=lambda kv: kv[1])]
    day_to_x = {d: i for i, d in enumerate(unique_days)}

    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    ax.grid(True, linestyle="--", alpha=0.3)

    hover_targets = []
    track_handles = []
    fault_handles = []

    # --- tracks (outline-only rectangles; NO fill)
    if not tracks.empty and len(day_to_x) > 0:
        tr = tracks.copy()

        sat_series = tr["satellite"].astype("string")
        missing = sat_series.isna() | sat_series.str.strip().eq("") | sat_series.str.strip().str.lower().isin({"none", "nan"})
        tr["satellite"] = sat_series.mask(missing, np.nan)

        named = tr.dropna(subset=["satellite"])
        sats = sorted(named["satellite"].unique().tolist())
        cmap = _color_map(sats)

        box_w = 0.70  # day column width (x±0.35)

        for sat in sats:
            grp = named[named["satellite"] == sat]
            color = cmap[sat]
            for _, row in grp.iterrows():
                x = day_to_x.get(row["day_label"])
                if x is None or pd.isna(row["y1"]) or pd.isna(row["y2"]):
                    continue
                y1, y2 = float(row["y1"]), float(row["y2"])
                if y2 <= y1:
                    continue

                rect = Rectangle(
                    (x - box_w / 2, y1),
                    box_w,
                    (y2 - y1),
                    fill=False,
                    edgecolor=color,
                    linewidth=1.5,
                    alpha=0.9,
                    zorder=2,
                )
                ax.add_patch(rect)

                # tooltips (best-effort; dt_start/dt_end may be missing after merges)
                start_s = pd.to_datetime(row.get("dt_start")).strftime("%H:%M") if pd.notna(row.get("dt_start")) else f"{y1:0.2f}h"
                end_s = pd.to_datetime(row.get("dt_end")).strftime("%H:%M") if pd.notna(row.get("dt_end")) else f"{y2:0.2f}h"
                text = f"Track: {sat}\nDay: {row['day_label']}\nStart–End: {start_s}–{end_s}"
                setattr(rect, "_hover_text", text)
                hover_targets.append((rect, text, False))

            track_handles.append(Patch(facecolor=color, alpha=0.25, label=sat))

        # unnamed tracks (gray outline)
        unnamed = tr[tr["satellite"].isna()]
        for _, row in unnamed.iterrows():
            x = day_to_x.get(row["day_label"])
            if x is None or pd.isna(row["y1"]) or pd.isna(row["y2"]):
                continue
            y1, y2 = float(row["y1"]), float(row["y2"])
            if y2 <= y1:
                continue

            rect = Rectangle(
                (x - box_w / 2, y1),
                box_w,
                (y2 - y1),
                fill=False,
                edgecolor=(0.5, 0.5, 0.5, 1),
                linewidth=1.2,
                alpha=0.8,
                zorder=2,
            )
            ax.add_patch(rect)

            start_s = pd.to_datetime(row.get("dt_start")).strftime("%H:%M") if pd.notna(row.get("dt_start")) else f"{y1:0.2f}h"
            end_s = pd.to_datetime(row.get("dt_end")).strftime("%H:%M") if pd.notna(row.get("dt_end")) else f"{y2:0.2f}h"
            text = f"Track Window\nDay: {row['day_label']}\nStart–End: {start_s}–{end_s}"
            setattr(rect, "_hover_text", text)
            hover_targets.append((rect, text, False))

    # --- faults
    if not faults.empty and len(day_to_x) > 0:
        for name, grp in faults.groupby("code_str", dropna=True):
            xs = grp["day_label"].map(day_to_x)
            keep = xs.notna() & grp["hour_float"].notna()
            xs = xs[keep].astype(float).values
            ys = grp.loc[keep, "hour_float"].astype(float).clip(lower=0, upper=23.999).values
            if len(xs) == 0:
                continue
            sc = ax.scatter(xs, ys, s=35, label=name, zorder=3)
            fault_handles.append(sc)

            texts = []
            for _, r in grp.loc[keep].iterrows():
                if pd.notna(r.get("dt")):
                    ts = pd.to_datetime(r["dt"]).strftime("%Y-%m-%d %H:%M")
                else:
                    ts = f"{r['day_label']} @ {float(r['hour_float']):.2f}h"
                texts.append(f"{name}\nTime: {ts}")
            setattr(sc, "_hover_texts", texts)
            hover_targets.append((sc, texts, True))

    # axes
    ax.set_xlim(-0.5, (len(unique_days) - 0.5) if unique_days else 0.5)
    ax.set_ylim(0, 24)
    ax.set_yticks(range(0, 25, 3))
    ax.set_ylabel("Time (24 Hour)")

    ax.set_xticks(range(len(unique_days)))
    ax.set_xticklabels(unique_days, rotation=45, ha="right")
    ax.set_xlabel("Day")
    ax.set_title(month_title or "DGS Fault & Track Map")

    # legends
    legend_y = -0.22
    all_h = fault_handles + track_handles
    if all_h:
        ax.legend(
            all_h, [h.get_label() for h in all_h],
            loc="upper center", bbox_to_anchor=(0.5, legend_y),
            ncol=min(6, max(1, len(all_h))), frameon=False
        )

    plt.subplots_adjust(bottom=0.28)
    plt.tight_layout()

    # hover
    if MPLCURSORS_AVAILABLE and hover_targets:
        artists = [a for (a, _, _) in hover_targets]
        cursor = mplcursors.cursor(artists, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            art = sel.artist
            if hasattr(art, "_hover_texts"):
                idx = sel.index
                texts = getattr(art, "_hover_texts", [])
                sel.annotation.set_text(texts[idx] if idx is not None and 0 <= idx < len(texts) else "Fault")
            elif hasattr(art, "_hover_text"):
                sel.annotation.set_text(getattr(art, "_hover_text"))
            else:
                sel.annotation.set_text("")

    return fig, ax


class DGSMapperPanel(ttk.Frame):
    """
    Same controls/behavior as the original Tk() App,
    but hosted inside a Frame so it can live inside a tab.
    """
    def __init__(self, master):
        super().__init__(master)
        self.files: List[str] = []

        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.title_var = tk.StringVar(value="DGS Fault & Track Map")
        ttk.Label(top, text="Plot Title:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.title_var, width=40).pack(side=tk.LEFT, padx=5)

        ttk.Button(top, text="Add Files", command=self.add_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Clear List", command=self.clear_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Plot", command=self.make_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Save Plot", command=self.save_plot).pack(side=tk.LEFT, padx=5)

        self.listbox = tk.Listbox(self, height=6)
        self.listbox.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.fig, self.ax = plt.subplots(figsize=(12, 6), dpi=120)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        if not MPLCURSORS_AVAILABLE:
            ttk.Label(
                self,
                text="Tip: install 'mplcursors' for hover tooltips:  pip install mplcursors",
                foreground="gray"
            ).pack(pady=(0, 8))

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select logs (CSV/XLSX/TXT/LOG)",
            filetypes=[
                ("All supported", "*.csv *.xlsx *.xls *.txt *.log"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("Text logs", "*.txt *.log"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        for p in paths:
            if p not in self.files:
                self.files.append(p)
                self.listbox.insert(tk.END, p)

    def clear_files(self):
        self.files = []
        self.listbox.delete(0, tk.END)

    def make_plot(self):
        if not self.files:
            messagebox.showwarning("No files", "Please add one or more files.")
            return

        faults_all, tracks_all, errors = [], [], []

        for path in self.files:
            try:
                df = load_any(path)
                f, t = extract_events(df)
                if not f.empty:
                    f = f.copy()
                    f["source"] = os.path.basename(path)
                    faults_all.append(f)
                if not t.empty:
                    t = t.copy()
                    t["source"] = os.path.basename(path)
                    tracks_all.append(t)
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")

        if errors:
            messagebox.showinfo("Some files skipped", "Issues encountered:\n" + "\n".join(errors))

        self.fig.clf()
        self.fig, self.ax = plot_fault_map(faults_all, tracks_all, month_title=self.title_var.get())
        self.canvas.figure = self.fig
        self.canvas.draw()

    def save_plot(self):
        if self.canvas is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            title="Save plot as"
        )
        if not path:
            return
        self.canvas.figure.savefig(path, bbox_inches="tight", dpi=200)
        messagebox.showinfo("Saved", f"Plot saved to:\n{path}")


# =============================================================================
# TAB 2: Antenna Fault Heatmap (logic kept the same; only GUI host changed)
# =============================================================================

# =========================
# Defaults
# =========================
MATCH_TOLERANCE_SEC = 5    # max allowed time difference between event and nearest metrics row
AZ_BIN_DEG = 5             # heatmap az bin width (degrees)
EL_BIN_DEG = 5             # heatmap el bin width (degrees)

# Filename patterns
METRICS_RE = re.compile(r"^Metrics_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}\.log\.csv$", re.IGNORECASE)
EVENTS_RE  = re.compile(r"^Events_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}\.log\.txt$", re.IGNORECASE)

# Events line pattern: timestamp at start
EVENT_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?),\s*(?P<msg>.*)$"
)

# Extract error code (only coded faults)
ERROR_CODE_RE = re.compile(r"\bError\s*code\s*(?P<code>\d{3,6})\b", re.IGNORECASE)


# =========================
# Helpers
# =========================

def parse_dt2(s: str) -> datetime:
    # NOTE: same logic as original, but renamed to avoid clashing with Tab1's parse_dt()
    if "." in s:
        main, frac = s.split(".", 1)
        frac = (frac + "000000")[:6]
        base = datetime.strptime(main, "%Y-%m-%d %H:%M:%S")
        return base.replace(microsecond=int(frac))
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

def to_polar_r(el_deg: float) -> float:
    # center=zenith (90), outer=horizon (0)
    return 90.0 - el_deg

def date_range_from_selection(
    year: int,
    month: str,
    day: str,
    use_range: bool = False,
    from_day: Optional[str] = None,
    to_day: Optional[str] = None,
) -> Tuple[datetime, datetime]:
    """
    Supports:
      - month=All => whole year
      - day=All   => whole month
      - day=DD    => specific day
      - use_range + from_day/to_day => day range within month (inclusive)
    """
    if month == "All":
        return datetime(year, 1, 1), datetime(year + 1, 1, 1)

    m = int(month)

    if use_range and from_day and to_day:
        d1 = int(from_day)
        d2 = int(to_day)
        if d2 < d1:
            d1, d2 = d2, d1  # auto-fix reversed input
        start = datetime(year, m, d1)
        end = datetime(year, m, d2) + timedelta(days=1)  # inclusive range
        return start, end

    if day == "All":
        start = datetime(year, m, 1)
        end = datetime(year + 1, 1, 1) if m == 12 else datetime(year, m + 1, 1)
        return start, end

    d = int(day)
    start = datetime(year, m, d)
    return start, start + timedelta(days=1)

def find_files_in_range(metrics_dir: Path, events_dir: Path, start: datetime, end: datetime) -> Tuple[List[Path], List[Path]]:
    metrics_files: List[Path] = []
    events_files: List[Path] = []

    if metrics_dir.exists():
        for p in metrics_dir.iterdir():
            m = METRICS_RE.match(p.name)
            if not m:
                continue
            d = datetime.strptime(m.group(1), "%Y-%m-%d")
            if start <= d < end:
                metrics_files.append(p)

    if events_dir.exists():
        for p in events_dir.iterdir():
            m = EVENTS_RE.match(p.name)
            if not m:
                continue
            d = datetime.strptime(m.group(1), "%Y-%m-%d")
            if start <= d < end:
                events_files.append(p)

    metrics_files.sort()
    events_files.sort()
    return metrics_files, events_files

def list_available_years(metrics_dir: Path, events_dir: Path) -> List[int]:
    years = set()
    if metrics_dir.exists():
        for p in metrics_dir.iterdir():
            m = METRICS_RE.match(p.name)
            if m:
                years.add(int(m.group(1)[:4]))
    if events_dir.exists():
        for p in events_dir.iterdir():
            m = EVENTS_RE.match(p.name)
            if m:
                years.add(int(m.group(1)[:4]))
    return sorted(years)

def list_available_dates_for_year(metrics_dir: Path, events_dir: Path, year: int) -> List[str]:
    dates = set()
    if metrics_dir.exists():
        for p in metrics_dir.iterdir():
            m = METRICS_RE.match(p.name)
            if m and m.group(1).startswith(f"{year:04d}-"):
                dates.add(m.group(1))
    if events_dir.exists():
        for p in events_dir.iterdir():
            m = EVENTS_RE.match(p.name)
            if m and m.group(1).startswith(f"{year:04d}-"):
                dates.add(m.group(1))
    return sorted(dates)

def load_metrics_csv(path: Path) -> Optional[pd.DataFrame]:
    """
    Metrics CSV may contain pre-header lines or may be malformed.
    We find a header line that starts with Time and read from there.

    If NO valid header is found, return None (skip file).

    Required columns:
      Time
      Antenna azimuth (deg)
      Antenna elevation (deg)
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    header_idx = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('"Time"') or s.startswith("Time,") or s.startswith("Time\t"):
            header_idx = i
            break

    if header_idx is None:
        return None  # skip invalid metrics file

    csv_text = "".join(lines[header_idx:])
    df = pd.read_csv(pd.io.common.StringIO(csv_text), engine="python")
    df.columns = [c.strip().strip('"') for c in df.columns]

    needed = {"Time", "Antenna azimuth (deg)", "Antenna elevation (deg)"}
    if not needed.issubset(set(df.columns)):
        return None  # skip malformed metrics file

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["Antenna azimuth (deg)"] = pd.to_numeric(df["Antenna azimuth (deg)"], errors="coerce")
    df["Antenna elevation (deg)"] = pd.to_numeric(df["Antenna elevation (deg)"], errors="coerce")

    df = df.dropna(subset=["Time", "Antenna azimuth (deg)", "Antenna elevation (deg)"]).sort_values("Time")
    if df.empty:
        return None

    df["Antenna azimuth (deg)"] = df["Antenna azimuth (deg)"] % 360.0
    df["Antenna elevation (deg)"] = df["Antenna elevation (deg)"].clip(0.0, 90.0)

    return df[["Time", "Antenna azimuth (deg)", "Antenna elevation (deg)"]].reset_index(drop=True)

@dataclass
class FaultEvent:
    time: datetime
    code: str
    msg: str

def load_fault_events_txt(path: Path) -> List[FaultEvent]:
    """
    Only returns lines that contain 'Error code ####' and have a timestamp at start.
    """
    out: List[FaultEvent] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            m = EVENT_LINE_RE.match(line)
            if not m:
                continue
            msg = m.group("msg").strip()

            code_m = ERROR_CODE_RE.search(msg)
            if not code_m:
                continue

            ts = parse_dt2(m.group("ts"))
            code = code_m.group("code")
            out.append(FaultEvent(ts, code, msg))
    return out

def match_faults_to_metrics(metrics: pd.DataFrame, faults: List[FaultEvent], tolerance_sec: int) -> pd.DataFrame:
    """
    Match each fault timestamp to nearest metrics timestamp to obtain az/el.
    """
    if metrics.empty or not faults:
        return pd.DataFrame(columns=["Time_event", "code", "msg", "Time_metrics", "az_deg", "el_deg"])

    df_ev = pd.DataFrame(
        [(f.time, f.code, f.msg) for f in faults],
        columns=["Time_event", "code", "msg"]
    ).sort_values("Time_event")

    df_met = metrics.rename(columns={
        "Time": "Time_metrics",
        "Antenna azimuth (deg)": "az_deg",
        "Antenna elevation (deg)": "el_deg",
    }).sort_values("Time_metrics")

    merged = pd.merge_asof(
        df_ev,
        df_met,
        left_on="Time_event",
        right_on="Time_metrics",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_sec),
    )

    return merged.dropna(subset=["Time_metrics", "az_deg", "el_deg"]).reset_index(drop=True)


class HeatmapPanel(ttk.Frame):
    """
    Same controls/behavior as the original Tk() App,
    but hosted inside a Frame so it can live inside a tab.
    """
    def __init__(self, master):
        super().__init__(master)

        self.root_dir: Optional[Path] = None
        self.events_dir: Optional[Path] = None
        self.metrics_dir: Optional[Path] = None

        self.year_var = tk.StringVar(value="")
        self.month_var = tk.StringVar(value="All")
        self.day_var = tk.StringVar(value="All")

        # range within month
        self.use_range_var = tk.BooleanVar(value=False)
        self.from_day_var = tk.StringVar(value="01")
        self.to_day_var = tk.StringVar(value="01")

        self.show_heatmap_var = tk.BooleanVar(value=True)
        self.show_fault_points_var = tk.BooleanVar(value=False)

        self.tolerance_var = tk.IntVar(value=MATCH_TOLERANCE_SEC)
        self.az_bin_var = tk.DoubleVar(value=AZ_BIN_DEG)
        self.el_bin_var = tk.DoubleVar(value=EL_BIN_DEG)

        self.status_var = tk.StringVar(value="Choose root folder containing Events/ and Metrics/.")
        self.details_var = tk.StringVar(value="Click a heatmap cell to see details here.")

        # For click inspection
        self._last_H = None
        self._last_az_edges = None
        self._last_el_edges = None
        self._last_matched = None

        self._build_ui()
        self._build_plot()

    def _build_ui(self):
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer)
        left.pack(side="left", fill="y")

        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        # 1) Folder selector
        lf_folder = ttk.LabelFrame(left, text="1) Root Folder", padding=10)
        lf_folder.pack(fill="x", pady=(0, 10))

        ttk.Button(lf_folder, text="Choose Root Folder", command=self.choose_root_folder).pack(fill="x")
        self.folder_label = ttk.Label(lf_folder, text="(none)")
        self.folder_label.pack(fill="x", pady=(6, 0))

        # 2) Date selector
        lf_date = ttk.LabelFrame(left, text="2) Date Selection", padding=10)
        lf_date.pack(fill="x", pady=(0, 10))

        ttk.Label(lf_date, text="Year:").pack(anchor="w")
        self.year_cb = ttk.Combobox(lf_date, textvariable=self.year_var, state="readonly", values=[])
        self.year_cb.pack(fill="x", pady=(0, 6))
        self.year_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_month_day_options())

        ttk.Label(lf_date, text="Month:").pack(anchor="w")
        self.month_cb = ttk.Combobox(
            lf_date, textvariable=self.month_var, state="readonly",
            values=["All"] + [f"{i:02d}" for i in range(1, 13)]
        )
        self.month_cb.pack(fill="x", pady=(0, 6))
        self.month_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_month_day_options())

        ttk.Label(lf_date, text="Day:").pack(anchor="w")
        self.day_cb = ttk.Combobox(lf_date, textvariable=self.day_var, state="readonly", values=["All"])
        self.day_cb.pack(fill="x", pady=(0, 4))

        ttk.Checkbutton(
            lf_date,
            text="Use date range within month",
            variable=self.use_range_var,
            command=self._toggle_range_controls
        ).pack(anchor="w", pady=(6, 2))

        range_row = ttk.Frame(lf_date)
        range_row.pack(fill="x")
        self._range_row = range_row

        ttk.Label(range_row, text="From:").pack(side="left")
        self.from_day_cb = ttk.Combobox(range_row, textvariable=self.from_day_var, state="readonly", width=6, values=[])
        self.from_day_cb.pack(side="left", padx=(4, 10))

        ttk.Label(range_row, text="To:").pack(side="left")
        self.to_day_cb = ttk.Combobox(range_row, textvariable=self.to_day_var, state="readonly", width=6, values=[])
        self.to_day_cb.pack(side="left", padx=(4, 0))

        self._toggle_range_controls()

        # 3) Options
        lf_opts = ttk.LabelFrame(left, text="3) Options", padding=10)
        lf_opts.pack(fill="x", pady=(0, 10))

        ttk.Checkbutton(lf_opts, text="Show heat map (frequency)", variable=self.show_heatmap_var).pack(anchor="w")
        ttk.Checkbutton(lf_opts, text="Show fault points (on top of heatmap)", variable=self.show_fault_points_var).pack(anchor="w")

        row = ttk.Frame(lf_opts)
        row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text="Match tol (sec):").pack(side="left")
        ttk.Entry(row, textvariable=self.tolerance_var, width=6).pack(side="left", padx=6)

        row2 = ttk.Frame(lf_opts)
        row2.pack(fill="x", pady=(6, 0))
        ttk.Label(row2, text="Az bin (deg):").pack(side="left")
        ttk.Entry(row2, textvariable=self.az_bin_var, width=6).pack(side="left", padx=6)
        ttk.Label(row2, text="El bin (deg):").pack(side="left")
        ttk.Entry(row2, textvariable=self.el_bin_var, width=6).pack(side="left", padx=6)

        # 4) Actions
        lf_actions = ttk.LabelFrame(left, text="4) Actions", padding=10)
        lf_actions.pack(fill="x", pady=(0, 10))

        ttk.Button(lf_actions, text="Load + Plot", command=self.load_and_plot).pack(fill="x")
        ttk.Button(lf_actions, text="Export PNG", command=self.export_png).pack(fill="x", pady=(6, 0))

        # Clicked details
        lf_details = ttk.LabelFrame(left, text="Clicked Cell Details", padding=10)
        lf_details.pack(fill="x", pady=(0, 10))
        ttk.Label(lf_details, textvariable=self.details_var, wraplength=360, justify="left").pack(fill="x")

        ttk.Label(left, textvariable=self.status_var, wraplength=360).pack(fill="x", pady=(6, 0))

        self.plot_frame = ttk.Frame(right)
        self.plot_frame.pack(fill="both", expand=True)

    def _toggle_range_controls(self):
        enabled = self.use_range_var.get()
        # Only enable range controls if month is not All
        if self.month_var.get() == "All":
            enabled = False
        state = "readonly" if enabled else "disabled"
        try:
            self.from_day_cb.configure(state=state)
            self.to_day_cb.configure(state=state)
        except Exception:
            pass

    def _build_plot(self):
        self.fig = plt.Figure(figsize=(7.5, 7.5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="polar")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._style_axes()
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)
        self.canvas.draw()

    def _style_axes(self):
        ax = self.ax
        ax.clear()
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)  # clockwise
        ax.set_rlim(0, 90)

        rings = [0, 15, 30, 45, 60, 75, 90]
        ax.set_yticks(rings)
        ax.set_yticklabels([f"{int(90-r)}°" for r in rings])
        ax.grid(True, alpha=0.35)
        ax.set_title("Fault Frequency Heat Map (Az/El)", pad=18)

    # -------- Folder + date --------
    def choose_root_folder(self):
        folder = filedialog.askdirectory(title="Select ROOT folder with Events/ and Metrics/")
        if not folder:
            return

        root = Path(folder)
        events = root / "Events"
        metrics = root / "Metrics"

        if not events.exists() or not metrics.exists():
            messagebox.showerror(
                "Folder structure not found",
                "Selected root folder must contain:\n"
                "  Events\\  (txt logs)\n"
                "  Metrics\\ (csv logs)\n\n"
                f"Selected: {root}"
            )
            return

        self.root_dir = root
        self.events_dir = events
        self.metrics_dir = metrics
        self.folder_label.config(text=str(root))

        years = list_available_years(metrics, events)
        if not years:
            self.year_cb["values"] = []
            self.year_var.set("")
            self.status_var.set("No matching Metrics_/Events_ files found.")
            return

        self.year_cb["values"] = [str(y) for y in years]
        self.year_var.set(str(years[-1]))
        self.month_var.set("All")
        self.day_var.set("All")
        self.use_range_var.set(False)
        self.refresh_month_day_options()

        self.status_var.set("Folder OK. Select Year/Month/Day (or range) then Load + Plot.")
        self.details_var.set("Click a heatmap cell to see details here.")

    def refresh_month_day_options(self):
        if not (self.metrics_dir and self.events_dir):
            return
        if not self.year_var.get().isdigit():
            return

        year = int(self.year_var.get())
        dates = list_available_dates_for_year(self.metrics_dir, self.events_dir, year)

        month = self.month_var.get()
        if month == "All":
            self.day_cb["values"] = ["All"]
            self.day_var.set("All")
            self._toggle_range_controls()
            return

        valid_days = sorted({d[8:10] for d in dates if d[5:7] == month})
        day_values = ["All"] + valid_days
        self.day_cb["values"] = day_values
        if self.day_var.get() not in day_values:
            self.day_var.set("All")

        # Populate range day selectors
        if valid_days:
            self.from_day_cb["values"] = valid_days
            self.to_day_cb["values"] = valid_days

            if self.from_day_var.get() not in valid_days:
                self.from_day_var.set(valid_days[0])
            if self.to_day_var.get() not in valid_days:
                self.to_day_var.set(valid_days[-1])

        self._toggle_range_controls()

    # -------- Load + plot --------
    def load_and_plot(self):
        if not (self.metrics_dir and self.events_dir):
            messagebox.showwarning("Missing folder", "Choose the root folder first.")
            return
        if not self.year_var.get().isdigit():
            messagebox.showwarning("Missing year", "Select a year.")
            return

        year = int(self.year_var.get())
        month = self.month_var.get()
        day = self.day_var.get()

        start, end = date_range_from_selection(
            year,
            month,
            day,
            use_range=self.use_range_var.get(),
            from_day=self.from_day_var.get(),
            to_day=self.to_day_var.get(),
        )

        metrics_files, events_files = find_files_in_range(self.metrics_dir, self.events_dir, start, end)

        if not metrics_files:
            messagebox.showwarning("No metrics", f"No metrics files found for range {start.date()} to {end.date()}.")
            return
        if not events_files:
            messagebox.showwarning("No events", f"No events files found for range {start.date()} to {end.date()}.")

        # Load metrics (skip invalid ones)
        metrics_dfs = []
        skipped_metrics = []

        for p in metrics_files:
            try:
                df = load_metrics_csv(p)
                if df is None or df.empty:
                    skipped_metrics.append(p.name)
                    continue
                metrics_dfs.append(df)
            except Exception:
                skipped_metrics.append(p.name)
                continue

        if not metrics_dfs:
            messagebox.showerror(
                "No valid metrics files",
                "All metrics files were skipped.\n\n"
                "Reason: No file contained a valid CSV header starting with 'Time'\n"
                "and required az/el columns."
            )
            return

        metrics = (
            pd.concat(metrics_dfs, ignore_index=True)
              .sort_values("Time")
              .reset_index(drop=True)
        )

        # Load coded faults
        faults_all: List[FaultEvent] = []
        for p in events_files:
            try:
                faults_all.extend(load_fault_events_txt(p))
            except Exception as e:
                messagebox.showerror("Events load failed", f"{p.name}\n\n{e}")
                return

        tol = int(self.tolerance_var.get())
        matched = match_faults_to_metrics(metrics, faults_all, tolerance_sec=tol)

        # Plot
        self._style_axes()

        # Reset click-inspection cache
        self._last_H = None
        self._last_az_edges = None
        self._last_el_edges = None
        self._last_matched = matched

        if self.show_heatmap_var.get():
            H, az_edges, el_edges = self._plot_heatmap_white_zero(matched)
            self._last_H = H
            self._last_az_edges = az_edges
            self._last_el_edges = el_edges

        if self.show_fault_points_var.get():
            self._plot_fault_points(matched)

        self.canvas.draw()

        extra = f" | Skipped metrics: {len(skipped_metrics)}" if skipped_metrics else ""
        self.status_var.set(
            f"Range: {start.date()} to { (end - timedelta(days=1)).date() } | "
            f"Metrics files: {len(metrics_files)} (loaded {len(metrics_dfs)}){extra} | "
            f"Events files: {len(events_files)} | "
            f"Coded faults: {len(faults_all)} | Matched: {len(matched)} (tol={tol}s)"
        )
        self.details_var.set("Click a heatmap cell to see details here.")

    def _plot_fault_points(self, matched: pd.DataFrame):
        if matched.empty:
            return
        theta = np.deg2rad(matched["az_deg"].to_numpy(dtype=float))
        r = np.array([to_polar_r(e) for e in matched["el_deg"].to_numpy(dtype=float)], dtype=float)
        self.ax.scatter(theta, r, s=22, alpha=0.9)

    def _plot_heatmap_white_zero(self, matched: pd.DataFrame):
        if matched.empty:
            return None, None, None

        az_bin = float(self.az_bin_var.get())
        el_bin = float(self.el_bin_var.get())
        az_bin = max(1.0, min(60.0, az_bin))
        el_bin = max(1.0, min(45.0, el_bin))

        az_edges = np.arange(0, 360 + az_bin, az_bin)
        el_edges = np.arange(0, 90 + el_bin, el_bin)

        az = matched["az_deg"].to_numpy(dtype=float)
        el = matched["el_deg"].to_numpy(dtype=float)

        H, az_e, el_e = np.histogram2d(az, el, bins=[az_edges, el_edges])

        # Mask zeros -> render as white
        H_masked = np.ma.masked_where(H == 0, H)

        theta_edges = np.deg2rad(az_e)
        r_edges_desc = 90.0 - el_e
        H_flip = np.flip(H_masked, axis=1)
        r_edges_asc = np.sort(r_edges_desc)

        T, R = np.meshgrid(theta_edges, r_edges_asc, indexing="ij")

        cmap = plt.cm.Reds.copy()
        cmap.set_bad(color="white")  # masked cells -> white

        self.ax.pcolormesh(
            T, R, H_flip,
            shading="auto",
            cmap=cmap,
            alpha=0.90
        )

        return H, az_edges, el_edges

    # -------- Click handling --------
    def on_plot_click(self, event):
        if event.inaxes != self.ax:
            return
        if self._last_H is None or self._last_az_edges is None or self._last_el_edges is None:
            return
        if self._last_matched is None or self._last_matched.empty:
            return
        if event.xdata is None or event.ydata is None:
            return

        theta = event.xdata
        r = event.ydata

        az = (np.rad2deg(theta) % 360.0)
        el = 90.0 - r

        az_edges = self._last_az_edges
        el_edges = self._last_el_edges

        iaz = np.searchsorted(az_edges, az, side="right") - 1
        iel = np.searchsorted(el_edges, el, side="right") - 1

        if iaz < 0 or iel < 0 or iaz >= len(az_edges) - 1 or iel >= len(el_edges) - 1:
            self.details_var.set("Clicked outside binned region.")
            return

        az_lo, az_hi = az_edges[iaz], az_edges[iaz + 1]
        el_lo, el_hi = el_edges[iel], el_edges[iel + 1]

        m = self._last_matched
        in_bin = m[
            (m["az_deg"] >= az_lo) & (m["az_deg"] < az_hi) &
            (m["el_deg"] >= el_lo) & (m["el_deg"] < el_hi)
        ].copy()

        count = int(self._last_H[iaz, iel]) if self._last_H is not None else len(in_bin)

        if in_bin.empty:
            self.details_var.set(
                f"Bin AZ[{az_lo:.0f}–{az_hi:.0f})°, EL[{el_lo:.0f}–{el_hi:.0f})°\n"
                f"Count: 0"
            )
            return

        code_counts = in_bin["code"].value_counts().to_dict()
        code_summary = ", ".join([f"{k}:{v}" for k, v in sorted(code_counts.items(), key=lambda kv: (-kv[1], kv[0]))])

        sample_n = 8
        samples = in_bin.sort_values("Time_event")[["Time_event", "code", "msg"]].head(sample_n)
        sample_lines = "\n".join([f"- {t} | {c} | {msg[:110]}" for t, c, msg in samples.to_records(index=False)])
        more = "" if len(in_bin) <= sample_n else f"\n(+{len(in_bin)-sample_n} more)"

        self.details_var.set(
            f"Bin AZ[{az_lo:.0f}–{az_hi:.0f})°, EL[{el_lo:.0f}–{el_hi:.0f})°\n"
            f"Count: {count}\n"
            f"Codes: {code_summary}\n"
            f"Sample:\n{sample_lines}{more}"
        )

    # -------- Export --------
    def export_png(self):
        out = filedialog.asksaveasfilename(
            title="Save plot as PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")]
        )
        if not out:
            return
        try:
            self.fig.savefig(out, dpi=220, bbox_inches="tight")
            self.status_var.set(f"Saved: {out}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))


# =============================================================================
# Combined Root Window
# =============================================================================

class CombinedApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Set window icon (PNG) from ./graphics/PhilSA_v1-01.png
        try:
            icon_path = Path(__file__).resolve().parent / "graphics" / "PhilSA_v1-01.png"
            if icon_path.exists():
                self._app_icon = tk.PhotoImage(file=str(icon_path))
                self.iconphoto(True, self._app_icon)
        except Exception:
            # Non-fatal: keep default Tk icon if loading fails
            pass
        self.title("DGS Tools — Combined GUI (Mapper + Heatmap)")
        self.geometry("1360x900")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        tab1 = DGSMapperPanel(nb)
        tab2 = HeatmapPanel(nb)

        nb.add(tab1, text="Fault & Track Mapper")
        nb.add(tab2, text="Antenna Heatmap (Az/El)")


def main():
    CombinedApp().mainloop()


if __name__ == "__main__":
    main()
