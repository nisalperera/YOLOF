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

# Create non-root user early (as root)
ARG USER=ubuntu
ARG UID=1000
ARG GID=$UID

RUN groupadd --gid $GID $USER && \
    useradd -s /bin/bash --uid $UID --gid $GID -m $USER && \
    apt-get update && \
    apt-get install -y sudo && \
    echo "$USER ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER && \
    chmod 0440 /etc/sudoers.d/$USER

# Install system packages as root (more efficient)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    curl \
    software-properties-common \
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
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Switch to non-root user for application setup
USER $USER
ARG HOME=/home/${USER}
ENV HOME=${HOME}
WORKDIR ${HOME}

# Install PyTorch (as non-root user)
SHELL ["/bin/bash", "-c"]
RUN IFS='.' read -r MAJOR MINOR _ <<< "$CUDA_VERSION" && \
    CUDA_VERSION_NO_DOT="${MAJOR}${MINOR}" && \
    python3 -m pip install --no-cache-dir \
        torch==${TORCH_VERSION}+cu${CUDA_VERSION_NO_DOT} \
        torchvision==${TORCHVISION_VERSION}+cu${CUDA_VERSION_NO_DOT} \
        --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION_NO_DOT}
SHELL ["/bin/sh", "-c"]

# Install Python dependencies from git repos
RUN python3 -m pip install --no-cache-dir 'git+https://github.com/JunnYu/mish-cuda.git' && \
    python3 -m pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# Replace pillow with pillow-simd
RUN python3 -m pip uninstall pillow -y && \
    CC="cc -mavx2" python3 -m pip install -U --force-reinstall pillow-simd

# Install requirements.txt BEFORE copying code (better caching)
COPY --chown=${UID}:${GID} requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application code (use --chown to avoid extra layer)
COPY --chown=${UID}:${GID} . ${HOME}/YOLOF

# Install YOLOF package
RUN python3 -m pip install --no-cache-dir -e ${HOME}/YOLOF


ARG WANDB_API_KEY
ENV WANDB_API_KEY=${WANDB_API_KEY}

# Only login if key is provided
RUN if [ -n "$WANDB_API_KEY" ]; then \
        python3 -m wandb login --relogin $WANDB_API_KEY; \
    fi

ENV PYTHONUNBUFFERED=1
