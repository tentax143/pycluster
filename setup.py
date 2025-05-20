from setuptools import setup, find_packages

setup(
    name="winray",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "flask",
        "requests",
        "typer[all]",
        "rich"
    ],
    entry_points={
        "console_scripts": [
            "winray=winray.cli:app"
        ]
    },
    author="godofconquest",
    description="Lightweight Ray-like distributed task scheduler for Windows clusters",
    python_requires='>=3.8',
)
