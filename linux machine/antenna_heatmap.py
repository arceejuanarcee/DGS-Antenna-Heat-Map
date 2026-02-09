#!/usr/bin/env python3
"""
Antenna Fault Heatmap GUI (Az/El Skyplot) — ERROR CODE ONLY + CLICK INSPECT
Legacy-friendly version (works on older Linux installs):

Includes:
- Skips invalid metrics files (no 'Time' header / missing az/el columns)
- White bins where count == 0
- Click a heatmap cell to show which fault codes are inside (with counts + samples)
- Date filters:
    * Whole year
    * Whole month
    * Specific day
    * Custom day range within a selected month (e.g., Sep 10–18)
- NO variable annotations, NO f-strings, NO dataclasses, NO pandas.merge_asof
  (so it runs on older Python/Pandas stacks often found in legacy Ubuntu boxes)

Folder structure:
  Root/
    Events/   -> Events_YYYY-MM-DD_*.log.txt
    Metrics/  -> Metrics_YYYY-MM-DD_*.log.csv

Install (Windows):
  pip install pandas numpy matplotlib

Install (Linux legacy, recommended):
  sudo apt-get install -y python3-numpy python3-pandas python3-matplotlib python3-tk

Run:
  python3 antenna_heatmap.py
"""

import re
import io
from pathlib import Path
from datetime import datetime, timedelta

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
MATCH_TOLERANCE_SEC = 5    # max allowed time difference between event and nearest metrics row
AZ_BIN_DEG = 5             # heatmap az bin width (degrees)
EL_BIN_DEG = 5             # heatmap el bin width (degrees)

# Filename patterns
METRICS_RE = re.compile(r"^metrics_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}\.log\.csv$", re.IGNORECASE)
EVENTS_RE  = re.compile(r"^events_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}\.log\.txt$", re.IGNORECASE)

# Events line pattern: timestamp at start
EVENT_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,6})?),\s*(?P<msg>.*)$"
)

# Extract error code (only coded faults)
ERROR_CODE_RE = re.compile(r"\bError\s*code\s*(?P<code>\d{3,6})\b", re.IGNORECASE)


# =========================
# Helpers
# =========================

def _set_philsa_icon(root_tk):
    """
    Set window icon to ./graphics/PhilSA_v1-01.png (PNG).
    Non-fatal if missing or unsupported (app still runs).
    """
    try:
        base = Path(__file__).resolve().parent
        icon_path = base / "graphics" / "PhilSA_v1-01.png"
        if icon_path.exists():
            img = tk.PhotoImage(file=str(icon_path))
            root_tk.iconphoto(True, img)
            # keep a reference so it doesn't get GC'ed
            root_tk._philsa_icon_ref = img
    except Exception:
        pass

def parse_dt(s):
    """Parse 'YYYY-MM-DD HH:MM:SS[.ffffff]'."""
    if "." in s:
        main, frac = s.split(".", 1)
        frac = (frac + "000000")[:6]
        base = datetime.strptime(main, "%Y-%m-%d %H:%M:%S")
        return base.replace(microsecond=int(frac))
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

def to_polar_r(el_deg):
    # center=zenith (90), outer=horizon (0)
    return 90.0 - el_deg

def date_range_from_selection(year, month, day, use_range=False, from_day=None, to_day=None):
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

def find_files_in_range(metrics_dir, events_dir, start, end):
    metrics_files = []
    events_files = []

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

def list_available_years(metrics_dir, events_dir):
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

def list_available_dates_for_year(metrics_dir, events_dir, year):
    dates = set()
    if metrics_dir.exists():
        for p in metrics_dir.iterdir():
            m = METRICS_RE.match(p.name)
            if m and m.group(1).startswith("{:04d}-".format(year)):
                dates.add(m.group(1))
    if events_dir.exists():
        for p in events_dir.iterdir():
            m = EVENTS_RE.match(p.name)
            if m and m.group(1).startswith("{:04d}-".format(year)):
                dates.add(m.group(1))
    return sorted(dates)

def load_metrics_csv(path):
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
    df = pd.read_csv(io.StringIO(csv_text), engine="python")
    df.columns = [c.strip().strip('"') for c in df.columns]

    needed = set(["Time", "Antenna azimuth (deg)", "Antenna elevation (deg)"])
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

def load_fault_events_txt(path):
    """
    Only returns lines that contain 'Error code ####' and have a timestamp at start.
    Returns list of dicts: {"time": datetime, "code": str, "msg": str}
    """
    out = []
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

            ts = parse_dt(m.group("ts"))
            code = code_m.group("code")
            out.append({"time": ts, "code": code, "msg": msg})
    return out

def _dt_to_ns(dt_obj):
    """datetime -> int nanoseconds since epoch (safe for sorting/search)."""
    return int(pd.Timestamp(dt_obj).value)

def match_faults_to_metrics(metrics_df, faults, tolerance_sec):
    """
    Legacy matching (no merge_asof): for each fault, choose nearest metrics row by time.
    """
    if metrics_df is None or metrics_df.empty or not faults:
        return pd.DataFrame(columns=["Time_event", "code", "msg", "Time_metrics", "az_deg", "el_deg"])

    mdf = metrics_df.copy()
    mdf = mdf.rename(columns={
        "Time": "Time_metrics",
        "Antenna azimuth (deg)": "az_deg",
        "Antenna elevation (deg)": "el_deg",
    }).sort_values("Time_metrics").reset_index(drop=True)

    mt_ns = mdf["Time_metrics"].apply(lambda x: int(pd.Timestamp(x).value)).values.astype(np.int64)

    rows = []
    tol_ns = int(tolerance_sec) * 1000000000

    for fe in faults:
        ev_t = fe["time"]
        ev_ns = _dt_to_ns(ev_t)

        idx = int(np.searchsorted(mt_ns, ev_ns))

        best_i = None
        best_d = None

        # candidate: idx (right neighbor)
        if idx < len(mt_ns):
            d = abs(mt_ns[idx] - ev_ns)
            best_i, best_d = idx, d

        # candidate: idx-1 (left neighbor)
        if idx - 1 >= 0:
            d2 = abs(mt_ns[idx - 1] - ev_ns)
            if (best_d is None) or (d2 < best_d):
                best_i, best_d = idx - 1, d2

        if best_i is None:
            continue
        if best_d > tol_ns:
            continue

        mr = mdf.iloc[best_i]
        rows.append((
            ev_t,
            fe["code"],
            fe["msg"],
            mr["Time_metrics"],
            float(mr["az_deg"]),
            float(mr["el_deg"]),
        ))

    return pd.DataFrame(rows, columns=["Time_event", "code", "msg", "Time_metrics", "az_deg", "el_deg"])


# =========================
# GUI App
# =========================

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        _set_philsa_icon(self)

        self.title("Antenna Fault Frequency Heat Map (Az/El) — Error Code Only")
        self.geometry("1280x860")

        self.root_dir = None
        self.events_dir = None
        self.metrics_dir = None

        self.year_var = tk.StringVar(value="")
        self.month_var = tk.StringVar(value="All")
        self.day_var = tk.StringVar(value="All")

        # Date range within month
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

        # For click inspection cache:
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
            values=["All"] + ["{:02d}".format(i) for i in range(1, 13)]
        )
        self.month_cb.pack(fill="x", pady=(0, 6))
        self.month_cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_month_day_options())

        ttk.Label(lf_date, text="Day:").pack(anchor="w")
        self.day_cb = ttk.Combobox(lf_date, textvariable=self.day_var, state="readonly", values=["All"])
        self.day_cb.pack(fill="x", pady=(0, 4))

        # Date range within month
        ttk.Checkbutton(
            lf_date,
            text="Use date range within month",
            variable=self.use_range_var,
            command=self._toggle_range_controls
        ).pack(anchor="w", pady=(6, 2))

        range_row = ttk.Frame(lf_date)
        range_row.pack(fill="x")

        ttk.Label(range_row, text="From:").pack(side="left")
        self.from_day_cb = ttk.Combobox(range_row, textvariable=self.from_day_var, state="disabled", width=6, values=[])
        self.from_day_cb.pack(side="left", padx=(4, 10))

        ttk.Label(range_row, text="To:").pack(side="left")
        self.to_day_cb = ttk.Combobox(range_row, textvariable=self.to_day_var, state="disabled", width=6, values=[])
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
        enabled = bool(self.use_range_var.get())
        if self.month_var.get() == "All":
            enabled = False
        state = "readonly" if enabled else "disabled"
        self.from_day_cb.configure(state=state)
        self.to_day_cb.configure(state=state)

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
        ax.set_yticklabels(["{}°".format(int(90-r)) for r in rings])
        ax.grid(True, alpha=0.35)
        ax.set_title("Fault Frequency Heat Map (Az/El)")

    # -------- Folder + date --------
    def choose_root_folder(self):
        folder = filedialog.askdirectory(title="Select ROOT folder with Events/ and Metrics/")
        if not folder:
            return

        root = Path(folder)
        events = None
        metrics = None

        candidates_events = [root / "Events", root / "events"]
        candidates_metrics = [root / "Metrics", root / "metrics"]

        for p in candidates_events:
            if p.exists() and p.is_dir():
                events = p
                break

        for p in candidates_metrics:
            if p.exists() and p.is_dir():
                metrics = p
                break

        if (events is None) or (metrics is None):
            messagebox.showerror(
                "Folder structure not found",
                "Selected root folder must contain:\n"
                " Events/ or events/ (txt logs)\n"
                " Metrics/ or metrics/ (csv logs) \n\n"
                "Selected: {}".format(str(root))
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

        valid_days = sorted(set([d[8:10] for d in dates if d[5:7] == month]))
        day_values = ["All"] + valid_days
        self.day_cb["values"] = day_values
        if self.day_var.get() not in day_values:
            self.day_var.set("All")

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
            use_range=bool(self.use_range_var.get()),
            from_day=self.from_day_var.get(),
            to_day=self.to_day_var.get(),
        )

        metrics_files, events_files = find_files_in_range(self.metrics_dir, self.events_dir, start, end)

        if not metrics_files:
            messagebox.showwarning("No metrics", "No metrics files found for range {} to {}.".format(start.date(), end.date()))
            return
        if not events_files:
            messagebox.showwarning("No events", "No events files found for range {} to {}.".format(start.date(), end.date()))

        # Load metrics (skip invalid ones)
        metrics_dfs = []
        skipped_metrics = []

        for p in metrics_files:
            try:
                df = load_metrics_csv(p)
                if (df is None) or df.empty:
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

        metrics = pd.concat(metrics_dfs, ignore_index=True).sort_values("Time").reset_index(drop=True)

        # Load coded faults
        faults_all = []
        for p in events_files:
            try:
                faults_all.extend(load_fault_events_txt(p))
            except Exception as e:
                messagebox.showerror("Events load failed", "{}\n\n{}".format(p.name, str(e)))
                return

        tol = int(self.tolerance_var.get())
        matched = match_faults_to_metrics(metrics, faults_all, tolerance_sec=tol)

        # Plot
        self._style_axes()

        self._last_H = None
        self._last_az_edges = None
        self._last_el_edges = None
        self._last_matched = matched

        if bool(self.show_heatmap_var.get()):
            H, az_edges, el_edges = self._plot_heatmap_white_zero(matched)
            self._last_H = H
            self._last_az_edges = az_edges
            self._last_el_edges = el_edges

        if bool(self.show_fault_points_var.get()):
            self._plot_fault_points(matched)

        self.canvas.draw()

        extra = ""
        if skipped_metrics:
            extra = " | Skipped metrics: {}".format(len(skipped_metrics))

        self.status_var.set(
            "Range: {} to {} | Metrics files: {} (loaded {}){} | Events files: {} | Coded faults: {} | Matched: {} (tol={}s)".format(
                start.date(),
                (end - timedelta(days=1)).date(),
                len(metrics_files),
                len(metrics_dfs),
                extra,
                len(events_files),
                len(faults_all),
                len(matched),
                tol
            )
        )
        self.details_var.set("Click a heatmap cell to see details here.")

    def _plot_fault_points(self, matched):
        if matched is None or matched.empty:
            return
        theta = np.deg2rad(matched["az_deg"].values.astype(float))
        r = np.array([to_polar_r(e) for e in matched["el_deg"].values.astype(float)], dtype=float)
        self.ax.scatter(theta, r, s=22, alpha=0.9)

    def _plot_heatmap_white_zero(self, matched):
        if matched is None or matched.empty:
            return None, None, None

        az_bin = float(self.az_bin_var.get())
        el_bin = float(self.el_bin_var.get())
        az_bin = max(1.0, min(60.0, az_bin))
        el_bin = max(1.0, min(45.0, el_bin))

        az_edges = np.arange(0, 360 + az_bin, az_bin)
        el_edges = np.arange(0, 90 + el_bin, el_bin)

        az = matched["az_deg"].values.astype(float)
        el = matched["el_deg"].values.astype(float)

        H, az_e, el_e = np.histogram2d(az, el, bins=[az_edges, el_edges])

        # Mask zeros -> render as white
        H_masked = np.ma.masked_where(H == 0, H)

        theta_edges = np.deg2rad(az_e)
        r_edges_desc = 90.0 - el_e
        H_flip = H_masked[:, ::-1]
        r_edges_asc = np.sort(r_edges_desc)

        T, R = np.meshgrid(theta_edges, r_edges_asc, indexing="ij")

        cmap = plt.get_cmap("Reds")
        try:
            cmap.set_bad("white")
        except Exception:
            pass

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
        if (self._last_H is None) or (self._last_az_edges is None) or (self._last_el_edges is None):
            return
        if (self._last_matched is None) or self._last_matched.empty:
            return
        if (event.xdata is None) or (event.ydata is None):
            return

        theta = event.xdata
        r = event.ydata

        az = (np.rad2deg(theta) % 360.0)
        el = 90.0 - r

        az_edges = self._last_az_edges
        el_edges = self._last_el_edges

        iaz = int(np.searchsorted(az_edges, az, side="right") - 1)
        iel = int(np.searchsorted(el_edges, el, side="right") - 1)

        if (iaz < 0) or (iel < 0) or (iaz >= len(az_edges) - 1) or (iel >= len(el_edges) - 1):
            self.details_var.set("Clicked outside binned region.")
            return

        az_lo, az_hi = az_edges[iaz], az_edges[iaz + 1]
        el_lo, el_hi = el_edges[iel], el_edges[iel + 1]

        m = self._last_matched
        in_bin = m[
            (m["az_deg"] >= az_lo) & (m["az_deg"] < az_hi) &
            (m["el_deg"] >= el_lo) & (m["el_deg"] < el_hi)
        ].copy()

        count = int(self._last_H[iaz, iel])

        if in_bin.empty:
            self.details_var.set(
                "Bin AZ[{:.0f}–{:.0f})°, EL[{:.0f}–{:.0f})°\nCount: 0".format(az_lo, az_hi, el_lo, el_hi)
            )
            return

        vc = in_bin["code"].value_counts()
        code_counts = dict(zip(vc.index.tolist(), vc.values.tolist()))
        code_summary = ", ".join(["{}:{}".format(k, code_counts[k]) for k in sorted(code_counts, key=lambda x: (-code_counts[x], x))])

        sample_n = 8
        samples = in_bin.sort_values("Time_event")[["Time_event", "code", "msg"]].head(sample_n)

        lines = []
        for row in samples.itertuples(index=False):
            t = row[0]
            c = row[1]
            msg = row[2]
            lines.append("- {} | {} | {}".format(t, c, msg[:110]))
        sample_lines = "\n".join(lines)

        more = ""
        if len(in_bin) > sample_n:
            more = "\n(+{} more)".format(len(in_bin) - sample_n)

        self.details_var.set(
            "Bin AZ[{:.0f}–{:.0f})°, EL[{:.0f}–{:.0f})°\nCount: {}\nCodes: {}\nSample:\n{}{}".format(
                az_lo, az_hi, el_lo, el_hi, count, code_summary, sample_lines, more
            )
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
            self.status_var.set("Saved: {}".format(out))
        except Exception as e:
            messagebox.showerror("Export failed", str(e))


def main():
    App().mainloop()

if __name__ == "__main__":
    main()
