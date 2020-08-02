from setuptools import setup, find_packages

setup(
    name='lookahead',
    version='0.0.1',
    packages=find_packages(),
    # install_requires=['scikit-learn', 'scipy', 'numpy', 'torch', 'gpytorch']
    install_requires=['scikit-learn', 'scipy', 'numpy', 'jaxlib', 'numpyro', 'torch', 'qmcpy']
)
