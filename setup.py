from setuptools import find_packages, setup


def get_requirements():
    """Get requirements from requirements.txt"""
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
    return requirements


def is_installed_detectron2():
    try:
        import detectron2
        return True
    except ImportError:
        return False


def is_installed_mish_cuda():
    try:
        import mish_cuda
        return True
    except ImportError:
        return False
    
    
install_requires = get_requirements()

if not is_installed_mish:
    print("Installing mish-cuda from GitHub")
    install_requires.append("mish-cuda @ git+https://github.com/JunnYu/mish-cuda.git")

if not is_installed_d2:
    print("Installing detectron2 from GitHub")
    install_requires.append("detectron2 @ git+https://github.com/facebookresearch/detectron2.git")
    
print(f"Installing following packages: {install_requires}")

setup(
    name="yolof",
    version="0.1.0",
    author="Nisal Perera",
    url="https://github.com/nisalperera/YOLOF",
    description="YOLOF: Implementation forked from github.com/chensnathan",
    packages=find_packages(exclude=("configs", "datasets")),
    install_requires=install_requires,
    python_requires=">=3.8"
)
