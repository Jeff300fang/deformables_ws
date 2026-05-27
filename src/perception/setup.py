from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "checkpoints"),
            glob("checkpoints/*"),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jeff',
    maintainer_email='jeff300fang@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sam_detection_front = perception.sam_detection_front:main',
            'sam_detection_back = perception.sam_detection_back:main',
            'tapnn_front = perception.tap_next_next_front:main',
            'tapnn_back = perception.tap_next_next_back:main',
            'front_fit_spline = perception.front_fit_spline:main',
            'back_fit_spline = perception.back_fit_spline:main',
            'rope_points_joint = perception.rope_points_joint:main',
            'perception_switch_monitor = perception.perception_switch_monitor:main'
        ],
    },
)
