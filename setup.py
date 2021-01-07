from setuptools import setup, find_packages

setup(
    name='lookahead',
    version='0.0.2',
    packages=find_packages(),
    install_requires=['scikit-learn', 'scipy', 'numpy', 'qmcpy']
)
