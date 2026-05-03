import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'arena_evaluation'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(where='.', include=[f'{package_name}*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'configs', 'benchmark'),
         glob('configs/benchmark/README.md')),
        (os.path.join('share', package_name, 'configs', 'benchmark', 'suites'),
         glob('configs/benchmark/suites/*.yaml')),
        (os.path.join('share', package_name, 'configs', 'benchmark', 'contests'),
         glob('configs/benchmark/contests/*.yaml')),
    ],
    install_requires=['setuptools'],
    extras_require={
        'test': ['pytest>=7'],
    },
    zip_safe=True,
    maintainer='NamTruongTran',
    maintainer_email='trannamtruong98@gmail.com',
    description='Record, evaluate, and plot navigational metrics to evaluate ROS navigation planners',
    license='BSD',
    entry_points={
        'console_scripts': [
        'record = arena_evaluation.data_recorder_node:main',
        'metrics = arena_evaluation.get_metrics:main',
        'benchmark = arena_evaluation.benchmark.runner:cli_main',
        'evaluation_cli = arena_evaluation.benchmark.cli:main',
        ],
    },
)
