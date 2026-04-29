from setuptools import find_packages, setup
import subprocess
import sys

def detect_cuda_version():
    """Detect CUDA version from nvcc --version"""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            output = result.stdout
            # Look for "release X.Y" in output
            for line in output.split('\n'):
                if 'release' in line:
                    # Extract version from line like "release 12.4, V12.4.99"
                    parts = line.split('release')[1].strip().split(',')[0].strip()
                    return parts
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return None

def get_numpy_version(pytorch_version):
    """
    Get appropriate numpy version based on PyTorch version.
    PyTorch < 2.3.1 requires numpy < 2.0.0
    PyTorch >= 2.3.1 supports numpy >= 2.0.0
    """
    # Parse PyTorch version
    major, minor, patch = map(int, pytorch_version.split('.'))
    
    # Create tuple for comparison (major, minor, patch)
    version_tuple = (major, minor, patch)
    target_version = (2, 3, 1)
    
    if version_tuple < target_version:
        return "numpy<2.0.0"
    else:
        return "numpy>=2.0.0"

def get_pytorch_and_torchvision_versions(cuda_version):
    """
    Map CUDA version to latest compatible PyTorch and torchvision versions.
    Returns (pytorch_version, torchvision_version, cuda_index)
    """
    cuda_to_versions = {
        "13.0": ("2.10.0", "0.25.0", "cu130"),
        "12.9": ("2.8.0", "0.23.0", "cu129"),
        "12.8": ("2.10.0", "0.25.0", "cu128"),
        "12.7": ("2.10.0", "0.25.0", "cu127"),
        "12.6": ("2.10.0", "0.25.0", "cu126"),
        "12.5": ("2.10.0", "0.25.0", "cu125"),
        "12.4": ("2.6.0", "0.21.0", "cu124"),
        "12.1": ("2.5.1", "0.20.1", "cu121"),
        "12.0": ("2.5.1", "0.20.1", "cu120"),
        "11.8": ("2.7.1", "0.22.1", "cu118"),
        "11.7": ("2.7.1", "0.22.1", "cu117"),
        "11.6": ("2.7.1", "0.22.1", "cu116"),
    }
    
    if cuda_version:
        # Match major.minor version
        for cuda_key in cuda_to_versions.keys():
            if cuda_version.startswith(cuda_key.split('.')[0] + '.' + cuda_key.split('.')[1]):
                return cuda_to_versions[cuda_key]
        
        # Try to match just major version
        major_version = cuda_version.split('.')[0]
        for cuda_key in cuda_to_versions.keys():
            if cuda_key.startswith(major_version):
                return cuda_to_versions[cuda_key]
    
    # Default to CPU version if CUDA not detected
    return ("2.5.0", "0.20.0", "cpu")

# Detect CUDA version
cuda_version = detect_cuda_version()
pytorch_version, torchvision_version, cuda_index = get_pytorch_and_torchvision_versions(cuda_version)
numpy_version = get_numpy_version(pytorch_version)

print(f"Detected CUDA Version: {cuda_version if cuda_version else 'Not found (will use CPU)'}")
print(f"Installing PyTorch {pytorch_version} and torchvision {torchvision_version} for {cuda_index}")
print(f"Installing numpy {numpy_version}")

# Build the index URL based on CUDA version
if cuda_index == "cpu":
    index_url = "https://download.pytorch.org/whl/cpu"
else:
    index_url = f"https://download.pytorch.org/whl/{cuda_index}"

setup(
    name="yolof",
    version="0.1.0",
    author="Chensnathan",
    url="https://github.com/chensnathan/YOLOF",
    description="Souped YOLOF: Exploring Model Soup for Object Detection",
    packages=find_packages(exclude=("configs", "datasets")),
    install_requires=[
        f"torch=={pytorch_version}",
        f"torchvision=={torchvision_version}",
        numpy_version,
    ],
    dependency_links=[
        index_url,
    ],
    python_requires=">=3.8"
)
