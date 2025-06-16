"""
Setup script for PyCluster
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Base requirements
install_requires = [
    "dask[complete]>=2023.1.0",
    "distributed>=2023.1.0",
    "psutil>=5.8.0",
    "requests>=2.25.0",
    "flask>=2.0.0",
    "flask-cors>=3.0.0",
]

# Optional requirements
extras_require = {
    'gpu': [
        'pynvml>=11.0.0',
        'nvidia-ml-py>=11.0.0',
    ],
    'llm': [
        'torch>=1.9.0',
        'transformers>=4.20.0',
        'accelerate>=0.20.0',
        'bitsandbytes>=0.39.0',
    ],
    'dev': [
        'pytest>=6.0.0',
        'pytest-asyncio>=0.18.0',
        'black>=22.0.0',
        'flake8>=4.0.0',
        'mypy>=0.950',
    ],
    'all': [
        'pynvml>=11.0.0',
        'nvidia-ml-py>=11.0.0',
        'torch>=1.9.0',
        'transformers>=4.20.0',
        'accelerate>=0.20.0',
        'bitsandbytes>=0.39.0',
    ]
}

setup(
    name="pycluster",
    version="0.2.0",
    author="PyCluster Development Team",
    author_email="support@pycluster.org",
    description="A Windows-based Python clustering package with LLM support and modern dashboard",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pycluster/pycluster",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "pycluster=pycluster.cli_enhanced:main",
            "pycluster-head=pycluster.cli:head_node_cli",
            "pycluster-worker=pycluster.cli:worker_node_cli",
            "pycluster-diagnose=pycluster.windows_fixes:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pycluster": [
            "*.md",
            "*.txt",
            "*.yaml",
            "*.yml",
        ],
    },
    keywords=[
        "distributed computing",
        "clustering",
        "dask",
        "windows",
        "llm",
        "gpu",
        "machine learning",
        "artificial intelligence",
        "dashboard",
        "monitoring"
    ],
    project_urls={
        "Bug Reports": "https://github.com/pycluster/pycluster/issues",
        "Source": "https://github.com/pycluster/pycluster",
        "Documentation": "https://pycluster.readthedocs.io",
    },
)

