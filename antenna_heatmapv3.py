#!/usr/bin/env python3
"""
Antenna Fault Heatmap GUI (Az/El Skyplot) — ERROR CODE ONLY + CLICK INSPECT
UPDATED VERSION:
- Full MM/DD/YYYY range selection
- Quick buttons (Last 7, Last 30, This Month, This Year)
- White bins where count == 0
- Click heatmap cell to inspect codes
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =========================
# Defaults
# =========================
MATCH_TOLERANCE_SEC = 5
AZ_BIN_DEG = 5
EL_BIN_DEG = 5

METRICS_RE = re.compile(r"^Metrics_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}\.log\.csv$", re.IGNORECASE)
EVENTS_RE  = re.compile(r"^Events_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}\.log\.txt$", re.IGNORECASE)

EVENT_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?),\s*(?P<msg>.*)$"
)

ERROR_CODE_RE = re.compile(r"\bError\s*code\s*(?P<code>\d{3,6})\b", re.IGNORECASE)


# =========================
# Helpers
# =========================

def parse_dt(s: str) -> datetime:
    if "." in s:
        main, frac = s.split(".", 1)
        frac = (frac + "000000")[:6]
        base = datetime.strptime(main, "%Y-%m-%d %H:%M:%S")
        return base.replace(microsecond=int(frac))
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def to_polar_r(el_deg: float) -> float:
    return 90.0 - el_deg


def date_range_from_full_dates(from_str: str, to_str: str) -> Tuple[datetime, datetime]:
    try:
        start = datetime.strptime(from_str, "%m/%d/%Y")
        end = datetime.strptime(to_str, "%m/%d/%Y")
    except ValueError:
        raise ValueError("Date format must be MM/DD/YYYY")

    if end < start:
        start, end = end, start

    end = end + timedelta(days=1)
    return start, end


def find_files_in_range(metrics_dir: Path, events_dir: Path,
                        start: datetime, end: datetime) -> Tuple[List[Path], List[Path]]:

    metrics_files = []
    events_files = []

    for p in metrics_dir.iterdir():
        m = METRICS_RE.match(p.name)
        if m:
            d = datetime.strptime(m.group(1), "%Y-%m-%d")
            if start <= d < end:
                metrics_files.append(p)

    for p in events_dir.iterdir():
        m = EVENTS_RE.match(p.name)
        if m:
            d = datetime.strptime(m.group(1), "%Y-%m-%d")
            if start <= d < end:
                events_files.append(p)

    metrics_files.sort()
    events_files.sort()
    return metrics_files, events_files


def load_metrics_csv(path: Path) -> Optional[pd.DataFrame]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('"Time"') or line.strip().startswith("Time,"):
            header_idx = i
            break

    if header_idx is None:
        return None

    csv_text = "".join(lines[header_idx:])
    df = pd.read_csv(pd.io.common.StringIO(csv_text))

    df.columns = [c.strip().strip('"') for c in df.columns]

    needed = {"Time", "Antenna azimuth (deg)", "Antenna elevation (deg)"}
    if not needed.issubset(set(df.columns)):
        return None

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["Antenna azimuth (deg)"] = pd.to_numeric(df["Antenna azimuth (deg)"], errors="coerce")
    df["Antenna elevation (deg)"] = pd.to_numeric(df["Antenna elevation (deg)"], errors="coerce")

    df = df.dropna().sort_values("Time")

    df["Antenna azimuth (deg)"] %= 360
    df["Antenna elevation (deg)"] = df["Antenna elevation (deg)"].clip(0, 90)

    return df.reset_index(drop=True)


@dataclass
class FaultEvent:
    time: datetime
    code: str
    msg: str


def load_fault_events_txt(path: Path) -> List[FaultEvent]:
    out = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = EVENT_LINE_RE.match(line.strip())
            if not m:
                continue
            code_m = ERROR_CODE_RE.search(m.group("msg"))
            if not code_m:
                continue
            ts = parse_dt(m.group("ts"))
            out.append(FaultEvent(ts, code_m.group("code"), m.group("msg")))
    return out


def match_faults_to_metrics(metrics: pd.DataFrame,
                            faults: List[FaultEvent],
                            tolerance_sec: int) -> pd.DataFrame:

    if metrics.empty or not faults:
        return pd.DataFrame()

    df_ev = pd.DataFrame([(f.time, f.code, f.msg)
                          for f in faults],
                         columns=["Time_event", "code", "msg"]).sort_values("Time_event")

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

    return merged.dropna()


# =========================
# GUI
# =========================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Antenna Fault Heat Map (Az/El)")
        self.geometry("1280x860")

        self.root_dir = None
        self.events_dir = None
        self.metrics_dir = None

        self.tolerance_var = tk.IntVar(value=MATCH_TOLERANCE_SEC)
        self.az_bin_var = tk.DoubleVar(value=AZ_BIN_DEG)
        self.el_bin_var = tk.DoubleVar(value=EL_BIN_DEG)

        self.status_var = tk.StringVar(value="Choose root folder.")

        self._build_ui()
        self._build_plot()

    # =========================
    # UI
    # =========================

    def _build_ui(self):
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer)
        left.pack(side="left", fill="y")

        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        # Folder
        ttk.Button(left, text="Choose Root Folder",
                   command=self.choose_root_folder).pack(fill="x")

        # Date Range
        ttk.Label(left, text="From (MM/DD/YYYY):").pack(anchor="w", pady=(10,0))
        self.from_date_var = tk.StringVar()
        ttk.Entry(left, textvariable=self.from_date_var).pack(fill="x")

        ttk.Label(left, text="To (MM/DD/YYYY):").pack(anchor="w", pady=(6,0))
        self.to_date_var = tk.StringVar()
        ttk.Entry(left, textvariable=self.to_date_var).pack(fill="x")

        # Quick Buttons
        ttk.Label(left, text="Quick Select:").pack(anchor="w", pady=(8,4))

        ttk.Button(left, text="Last 7 Days",
                   command=self.set_last_7_days).pack(fill="x")
        ttk.Button(left, text="Last 30 Days",
                   command=self.set_last_30_days).pack(fill="x")
        ttk.Button(left, text="This Month",
                   command=self.set_this_month).pack(fill="x")
        ttk.Button(left, text="This Year",
                   command=self.set_this_year).pack(fill="x")

        ttk.Button(left, text="Load + Plot",
                   command=self.load_and_plot).pack(fill="x", pady=(10,0))

        ttk.Label(left, textvariable=self.status_var,
                  wraplength=350).pack(fill="x", pady=10)

        self.plot_frame = right

    # =========================
    # Quick Date Presets
    # =========================

    def _set_date_fields(self, start, end):
        self.from_date_var.set(start.strftime("%m/%d/%Y"))
        self.to_date_var.set(end.strftime("%m/%d/%Y"))

    def set_last_7_days(self):
        today = datetime.today()
        self._set_date_fields(today - timedelta(days=6), today)

    def set_last_30_days(self):
        today = datetime.today()
        self._set_date_fields(today - timedelta(days=29), today)

    def set_this_month(self):
        today = datetime.today()
        self._set_date_fields(today.replace(day=1), today)

    def set_this_year(self):
        today = datetime.today()
        self._set_date_fields(today.replace(month=1, day=1), today)

    # =========================
    # Folder
    # =========================

    def choose_root_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        root = Path(folder)
        self.root_dir = root
        self.events_dir = root / "Events"
        self.metrics_dir = root / "Metrics"
        self.status_var.set(f"Loaded folder: {root}")

    # =========================
    # Plot
    # =========================

    def _build_plot(self):
        self.fig = plt.Figure(figsize=(7.5,7.5))
        self.ax = self.fig.add_subplot(111, projection="polar")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_and_plot(self):
        if not self.metrics_dir:
            messagebox.showwarning("Select folder first")
            return

        try:
            start, end = date_range_from_full_dates(
                self.from_date_var.get(),
                self.to_date_var.get()
            )
        except Exception as e:
            messagebox.showerror("Invalid Date", str(e))
            return

        metrics_files, events_files = find_files_in_range(
            self.metrics_dir, self.events_dir, start, end
        )

        metrics = pd.concat(
            [load_metrics_csv(p) for p in metrics_files if load_metrics_csv(p) is not None],
            ignore_index=True
        )

        faults = []
        for p in events_files:
            faults.extend(load_fault_events_txt(p))

        matched = match_faults_to_metrics(metrics, faults,
                                          self.tolerance_var.get())

        self.ax.clear()
        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1)
        self.ax.set_rlim(0,90)

        if not matched.empty:
            az = matched["az_deg"].to_numpy()
            el = matched["el_deg"].to_numpy()

            H, az_edges, el_edges = np.histogram2d(
                az, el,
                bins=[
                    np.arange(0, 360 + self.az_bin_var.get(), self.az_bin_var.get()),
                    np.arange(0, 90 + self.el_bin_var.get(), self.el_bin_var.get())
                ]
            )

            H_masked = np.ma.masked_where(H == 0, H)

            T, R = np.meshgrid(
                np.deg2rad(az_edges),
                90 - el_edges,
                indexing="ij"
            )

            cmap = plt.cm.Reds.copy()
            cmap.set_bad("white")

            self.ax.pcolormesh(T, R, np.flip(H_masked,1),
                               shading="auto", cmap=cmap)

        self.canvas.draw()
        self.status_var.set(f"Matched faults: {len(matched)}")


def main():
    App().mainloop()


if __name__ == "__main__":
    main()