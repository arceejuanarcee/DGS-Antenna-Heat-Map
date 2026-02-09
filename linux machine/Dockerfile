FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /src

# System libs for Tkinter + Matplotlib runtime
RUN apt-get update && apt-get install -y \
    wget bzip2 ca-certificates \
    binutils \
    libx11-6 libxext6 libxrender1 libsm6 libice6 \
    libgl1-mesa-glx \
    tk \
    && rm -rf /var/lib/apt/lists/*


# Install Miniforge3 (conda-forge) - older installer that works with glibc 2.27
# Pinning to a known good release instead of "latest"
RUN wget -q https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Miniforge3-24.3.0-0-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

ENV PATH="/opt/conda/bin:$PATH"

# Create env with Tk + deps, then install PyInstaller
RUN conda install -y python=3.10 numpy pandas matplotlib tk pip && \
    pip install --no-cache-dir pyinstaller && \
    conda clean -afy

CMD ["bash"]
