from setuptools import setup

setup(
    name='fuzz1ng',
    version='0.0.1',
    install_requires=[
    ],
    packages=[
        'genetic',
        'utils',
    ],
    package_data={
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'genetic_fuzzer=genetic.fuzzer:run',
            'afl_dump=utils.afl:dump',
        ],
    },
)
