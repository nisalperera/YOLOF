ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

LABEL maintainer="Nisal Perera"

ENV CUDNN_VERSION=9.3.0.77
ENV FORCE_CUDA="1"
ENV CUDA_CACHE_DISABLE="1"
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0 12.0"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CUDA_VERSION=${CUDA_VERSION}
ENV TORCH_VERSION=2.8.0
ENV TORCHVISION_VERSION=0.23.0

ARG USER=ubuntu
ENV USER=$USER
ARG UID=1000
ARG GID=$UID
RUN if ! id -u $UID >/dev/null 2>&1; then \
        groupadd --gid $GID $USER && \
        useradd -s /bin/bash --uid $UID --gid $GID -m $USER; \
    fi
# Add sudo support for the non-root user
RUN apt-get update && \
    apt-get install -y sudo && \
    echo "$USER ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER && \
    chmod 0440 /etc/sudoers.d/$USER

# Switch from root to user
USER $USER

ARG HOME=/home/${USER}
ENV HOME=${HOME}

WORKDIR ${HOME}

RUN sudo apt-get update && sudo apt-get install -y \
    wget \
    gnupg \
    curl \
    software-properties-common

# Timezone non-interactive
# RUN sudo echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections \
#     && sudo apt-get update && sudo apt-get install -y tzdata \
#     && sudo rm -rf /var/lib/apt/lists/*

# Core packages
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    git \
    python3-dev \
    python3-pip \
    libffi-dev \
    cmake \
    gcc \
    g++ \
    ffmpeg \
    x264 \
    libx264-dev \
    pkg-config \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    libjpeg-turbo8-dev \
    zlib1g-dev \
    libtiff5-dev \
    liblcms2-dev \
    libfreetype6-dev \
    libwebp-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libopenjp2-7-dev \
    libraqm0 \
    openexr \
    libatlas-base-dev \
    libtbb2 \
    libtbb-dev \
    libdc1394-dev \
    openssh-client \
    ninja-build \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

# PyTorch 2.8 + cu126
SHELL ["/bin/bash", "-c"]
RUN IFS='.' read -r MAJOR MINOR _ <<< "$CUDA_VERSION" && \
    CUDA_VERSION_NO_DOT="${MAJOR}${MINOR}" && \
    python3 -m pip install \
        torch==${TORCH_VERSION}+cu${CUDA_VERSION_NO_DOT} \
        torchvision==${TORCHVISION_VERSION}+cu${CUDA_VERSION_NO_DOT} \
        --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION_NO_DOT}
SHELL ["/bin/sh", "-c"]

# mish-cuda, pillow-simd
RUN python3 -m pip install 'git+https://github.com/JunnYu/mish-cuda.git'
RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN python3 -m pip uninstall pillow -y
RUN CC="cc -mavx2" python3 -m pip install -U --force-reinstall pillow-simd

COPY . $HOME/YOLOF
RUN sudo chown -R ${USER}:${USER} $HOME/YOLOF
RUN python3 -m pip install -e ${HOME}/YOLOF

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

ENV PYTHONUNBUFFERED=1