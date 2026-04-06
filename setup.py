from setuptools import find_packages, setup
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [2, 3], "Requires PyTorch >= 2.0"

setup(
    name="yolof",
    version="0.1.0",
    author="Chensnathan",
    url="https://github.com/chensnathan/YOLOF",
    description="Souped YOLOF: Exploring Model Soup for Object Detection",
    packages=find_packages(exclude=("configs", "datasets")),
    python_requires=">=3.6"
)
