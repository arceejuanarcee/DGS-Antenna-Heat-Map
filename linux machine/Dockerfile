FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /src

# Basic deps + Tk + X11 libs needed by Tkinter/Matplotlib
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget ca-certificates \
    tk \
    python3-tk \
    libx11-6 libxext6 libxrender1 libsm6 libice6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10 from deadsnakes (includes shared lib)
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-distutils python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# Pip for Python 3.10
RUN wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py && \
    python3.10 /tmp/get-pip.py && \
    rm /tmp/get-pip.py

# Install python deps + pyinstaller
RUN python3.10 -m pip install --no-cache-dir --upgrade pip && \
    python3.10 -m pip install --no-cache-dir pyinstaller numpy pandas matplotlib

CMD ["bash"]
