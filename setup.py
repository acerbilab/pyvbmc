from setuptools import find_packages, setup

setup(
    name="pyvbmc",
    version="0.1.0",
    description="Variational Bayesian Monte Carlo",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pytest",
        "numpydoc",
        "cma",
        "corner",
        "imageio",
        "kaleido",
        "myst_nb>=0.17.1",
        "numpydoc>=1.2.1",
        "sphinx>=4.5.0",
        "sphinx-book-theme>=0.3.3",
        "pylint",
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "pytest-rerunfailures",
    ],
)
