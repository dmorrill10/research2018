from setuptools import setup, find_packages

setup(
    name='robust_offline_contextual_bandits',
    version='0.0.1',
    license='',
    packages=find_packages(),
    install_requires=['setuptools >= 20.2.2'],
    tests_require=['pytest', 'pytest-cov'],
    setup_requires=['pytest-runner'],
)
