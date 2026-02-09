FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /src

# System libs for Tkinter + Matplotlib runtime
RUN apt-get update && apt-get install -y \
    wget bzip2 ca-certificates \
    libx11-6 libxext6 libxrender1 libsm6 libice6 \
    libgl1-mesa-glx \
    tk \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (comes with shared libpython)
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

# Create env with Tk + deps
RUN conda install -y python=3.10 numpy pandas matplotlib tk pip && \
    pip install --no-cache-dir pyinstaller && \
    conda clean -afy

CMD ["bash"]
