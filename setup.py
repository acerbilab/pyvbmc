from setuptools import setup, find_packages

setup(name='pyvbmc',
      version='0.1.0',
      description='Variational Bayesian Monte Carlo',
      packages=find_packages(),
      install_requires=['numpy',
                        'scipy',
                        'matplotlib',
                        'pytest',
                        'sphinx',
                        'numpydoc',
                        'cma'],
     )