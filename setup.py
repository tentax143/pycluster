from setuptools import setup, find_packages

setup(
    name='winray',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'flask',
        'psutil',
        'gputil',
        # add other deps here
    ],
    entry_points={
        'console_scripts': [
            'winray-head=winray.head:main',
            'winray-worker=winray.worker:main',
        ],
    },
    author='Your Name',
    description='WinRay distributed task scheduling package',
    python_requires='>=3.7',
)
