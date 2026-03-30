apt update && apt install -y --no-install-recommends \
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
    ninja-build

python3 -m pip install 'git+https://github.com/JunnYu/mish-cuda.git'
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

wget https://raw.githubusercontent.com/nisalperera/YOLOF/refs/heads/hyperparam_sweep/requirements.txt

python3 -m pip install -r requirements.txt