#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import re

from setuptools import find_packages, setup

with io.open("torch_kitti/__init__.py", "rt", encoding="utf8") as f:
    searched = re.search(r"__version__ = \"(.*?)\"", f.read())
    if searched is None:
        raise ImportError("Could not find __version__ in torch_kitti/__init__.py")
    else:
        version = searched.group(1)

setup(
    name="torch-kitti",
    version=version,
    author="Andrea Conti",
    author_email="andrea.conti@tutanota.com",
    description="utilities and pytorch datasets for the KITTI Vision Benchmark Suite",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": ["torch_kitti = torch_kitti.scripts.torch_kitti:main"]
    },
    package_data={"": ["*.txt", "*.pkl"]},
    install_requires=[
        "torch >= 1.5.0",
        "torchvision >= 0.6.0",
        "setuptools >= 46.3.0",
        "Pillow > 7.1.0",
        "requests >= 2.24.0",
        "tqdm >= 4.51.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "pre-commit",
            "mypy",
            "flake8",
            "black",
            "isort",
            "tox",
        ],
        "doc": ["sphinx", "sphinx-rtd-theme"],
    },
)
