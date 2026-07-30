#!/usr/bin/env python3
"""Packaging for projector-distortion-framework."""

import os

from setuptools import find_packages, setup

_HERE = os.path.dirname(os.path.abspath(__file__))


def _read(name):
    path = os.path.join(_HERE, name)
    with open(path, encoding="utf-8") as f:
        return f.read()


def _requirements():
    reqs = []
    for line in _read("requirements.txt").splitlines():
        line = line.split("#", 1)[0].strip()
        # extras live in `extras_require`, not the base install
        if line and not line.startswith("-"):
            reqs.append(line)
    return reqs


setup(
    name="projector-distortion-framework",
    version="0.1.0",
    description="Remove projector light from camera captures, then compare object "
                "detection before and after restoration.",
    long_description=_read("README.md"),
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(include=["projector_distortion", "projector_distortion.*"]),
    package_data={"projector_distortion": ["configs/*.yaml"]},
    install_requires=_requirements(),
    extras_require={
        "yolo": ["ultralytics>=8.0"],
        "train": ["pytorch-msssim>=1.0", "pandas>=1.3", "matplotlib>=3.4"],
        "live": ["screeninfo>=0.8"],          # non-Windows only; Windows uses Win32
        "test": ["pytest>=7.0"],
        "all": ["ultralytics>=8.0", "pytorch-msssim>=1.0", "pandas>=1.3",
                "matplotlib>=3.4", "screeninfo>=0.8", "pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "pdf-demo=projector_distortion.cli:demo_main",
            "pdf-evaluate=projector_distortion.cli:evaluate_main",
            "pdf-train=projector_distortion.cli:train_main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
