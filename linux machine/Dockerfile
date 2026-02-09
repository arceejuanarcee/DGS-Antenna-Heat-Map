# manylinux2014 has old glibc compatibility (good for Mint 18.1)
FROM quay.io/pypa/manylinux2014_x86_64

WORKDIR /src

# Use the bundled Python 3.10 in manylinux
ENV PYBIN=/opt/python/cp310-cp310/bin

# Install build tools and X11/Tk runtime libs needed by matplotlib+tkinter packaging
RUN yum install -y \
    tk \
    xorg-x11-server-Xvfb \
    libX11 libXext libXrender libSM libICE mesa-libGL \
    && yum clean all

# Install Python packages
RUN ${PYBIN}/python -m pip install --upgrade pip && \
    ${PYBIN}/pip install pyinstaller numpy pandas matplotlib

CMD ["/bin/bash"]
