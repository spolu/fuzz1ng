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
            'transformer_trainer=transformer.trasnformer:train',
            'runs_db_eval=utils.runs_db:eval',
            'afl_dump=utils.afl:dump',
        ],
    },
)
