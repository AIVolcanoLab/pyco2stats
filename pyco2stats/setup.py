# setup.py

from setuptools import setup, find_packages

setup(
    name="PyCO2stats",  
    version="0.1.0",  
    description="A package for statistical analysis and plotting in the context of CO2 data",
    author="Your Name",
    author_email="maurizio.petrelli@unipg.it",
    url="https://github.com/yourusername/PyCO2stats",  # URL to the package's repository
    packages=find_packages(exclude=["tests"]),  # Automatically find packages except the tests directory
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.4.0",
        "matplotlib>=3.1.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "flake8>=3.8"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
