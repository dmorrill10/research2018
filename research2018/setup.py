from setuptools import setup, find_packages

setup(
    name='research2018',
    version='0.0.1',
    license='',
    packages=find_packages(),
    install_requires=[
        'setuptools >= 20.2.2',
        'tensorflow >= 2.0',
        # Not on pip:
        # 'tf-contextual_prediction_with_expert_advice',
        # 'tf-kofn_robust_policy_optimization',
    ],
    tests_require=['pytest', 'pytest-cov'],
    setup_requires=['pytest-runner'],
)
