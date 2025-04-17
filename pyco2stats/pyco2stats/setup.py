from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent.resolve()
long_description = (this_directory / "README.md").read_text()

setup(
    name="pyco2stats",
    version="0.1.0",
    description="A package for statistical analysis and plotting in the context of CO2 data",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Maurizio Petrelli, Alessandra Ariano",
    author_email="maurizio.petrelli@unipg.it",
    url="https://github.com/yourusername/pyco2stats",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "statsmodels",
        "astropy",
        "torch",
        "scikit-learn",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "flake8>=3.8",
            "sphinx>=3.0",
            "sphinx_rtd_theme"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
