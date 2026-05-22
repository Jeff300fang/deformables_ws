from setuptools import find_packages, setup

from glob import glob
import os

package_name = 'manipulator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jeffreyfang',
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
            "get_end_effector_pose = manipulator.get_end_effector_pose:main",
            "move_to_pose = manipulator.move_to_pose:main",
            "arm_driver = manipulator.arm_driver:main",
            "left_grip = manipulator.left_grip:main",
            "right_grip = manipulator.right_grip:main",
        ],
    },
)
