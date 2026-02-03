# Antenna Fault Frequency Heat Map (Az/El) — Linux Run Guide
Author: Arcee Juan, SRS1 - SMCOD

This tool generates an **Azimuth/Elevation skyplot heat map** showing **where antenna fault error codes most frequently occur**, based on Events logs matched to Metrics az/el data.

---

## 1) Required Folder Structure

Select a **root folder** that contains the following subfolders (case-insensitive):

ROOT/
  events/    or Events/     -> Events_YYYY-MM-DD_HH-MM-SS.log.txt  
  metrics/   or Metrics/    -> Metrics_YYYY-MM-DD_HH-MM-SS.log.csv  

IMPORTANT:
- Select the ROOT folder only.
- Do NOT select the events/ or metrics/ folder directly.

---

## 2) Linux Dependencies (System Packages)

For Debian / Ubuntu systems:

sudo apt-get update
sudo apt-get install -y python3 python3-tk python3-numpy python3-pandas python3-matplotlib

Notes:
- python3-tk is required for the GUI.
- The script is compatible with older system Python stacks.

---

## 3) Run the Application

From the directory containing the script:

/usr/bin/python3 antenna_heatmapv2.py

Or using an absolute path:

/usr/bin/python3 /full/path/to/antenna_heatmapv2.py

---

## 4) How to Use

1. Launch the script.
2. Click **Choose Root Folder** and select the folder containing events/ and metrics/.
3. Choose the desired date scope:
   - Whole year
   - Whole month
   - Specific day
   - Custom date range within a month
4. Click **Load + Plot**.
5. Click on any heatmap cell to view:
   - Fault count
   - Fault code breakdown
   - Sample fault messages with timestamps
6. Use **Export PNG** to save the visualization.

---

## 5) Heat Map Interpretation

- White cells: no faults in that az/el bin
- Light red: low fault frequency
- Dark red: high fault concentration
- Optional overlay of individual fault points can be enabled

---

## 6) Troubleshooting

Folder structure error:
- Ensure the selected root folder contains BOTH events/ and metrics/.

GUI does not open:
- Install Tkinter:
  sudo apt-get install -y python3-tk

Library import errors:
- Ensure system packages are installed:
  sudo apt-get install -y python3-numpy python3-pandas python3-matplotlib

Verify system Python:

/usr/bin/python3 --version
/usr/bin/python3 -c "import numpy, pandas, matplotlib; print('OK')"

---

## 7) Purpose

This tool is intended for:
- Visualizing recurring antenna fault regions
- Identifying azimuth/elevation hotspots
- Supporting troubleshooting, maintenance planning, and root-cause analysis
