from setuptools import setup

setup(
    name='fuzz1ng',
    version='0.0.1',
    install_requires=[
    ],
    packages=[
        'database',
    ],
    package_data={
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'test_fuzz1ng=database.test:run',
        ],
    },
)
