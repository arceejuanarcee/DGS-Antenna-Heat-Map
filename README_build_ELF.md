# Author: Arcee T. Juan
# Building the ELF Executable (Linux)

This guide explains how to build the standalone ELF executable for the
**Ground Station Antenna Performance Analysis Tool** using PyInstaller.

------------------------------------------------------------------------

## Prerequisites

Make sure the target Linux machine has:

-   Python 3.10+
-   pip
-   build tools (if not installed, run:
    `sudo apt install build-essential`)

------------------------------------------------------------------------

## Create a Virtual Environment (Recommended)

``` bash
python3 -m venv venv
source venv/bin/activate
```

------------------------------------------------------------------------

## Install Required Python Packages

Inside the virtual environment:

``` bash
pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller
```

If you do not have a requirements.txt, install manually:

``` bash
pip install numpy pandas matplotlib pillow
```

------------------------------------------------------------------------

## Build the ELF File

From the project root (where your main script is located):

``` bash
pyinstaller --onefile --clean --add-data "graphics:graphics" --hidden-import=PIL._tkinter_finder --hidden-import=PIL.ImageTk --hidden-import=tkinter --hidden-import=_tkinter --name antenna_monitoring antenna_monitoringv2.py
```

After completion, the executable will be located in:

    dist/antenna_monitoring

------------------------------------------------------------------------

## Verify It Is an ELF File

Run:

``` bash
file dist/antenna_monitoring
```

Expected output should include:

    ELF 64-bit LSB executable

------------------------------------------------------------------------

## Run the Application

From the terminal:

``` bash
./dist/antenna_monitoring
```

If needed, make it executable:

``` bash
chmod +x dist/antenna_monitoring
```

------------------------------------------------------------------------

## Optional: Clean Previous Builds

Before rebuilding:

``` bash
rm -rf build dist *.spec
```

------------------------------------------------------------------------

## Notes

-   The build must be performed on the target Linux system (or a system
    with compatible architecture and GLIBC version).
-   Linux is case-sensitive. Ensure folders are named exactly:
    -   `Events/`
    -   `Metrics/`
-   If Tkinter-related errors occur, install:

``` bash
sudo apt install python3-tk
```

------------------------------------------------------------------------

✅ After these steps, you will have a portable ELF executable ready to
run on your Linux machine.
