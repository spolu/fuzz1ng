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
            'genetic_energy_fuzzer=genetic.energy:fuzz',
            'genetic_simple_fuzzer=genetic.simple:fuzz',
            'transformer_trainer=transformer.transformer:train',
            'runs_db_eval=utils.runs_db:eval',
            'afl_dump=utils.afl:dump',
        ],
    },
)
