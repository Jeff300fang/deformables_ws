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
            "left_get_end_effector_pose = manipulator.left_get_end_effector_pose:main",
            "left_move_to_pose = manipulator.left_move_to_pose:main",
            "left_arm_driver = manipulator.left_arm_driver:main",
            "left_grip = manipulator.left_grip:main",
            "right_grip = manipulator.right_grip:main",
            "right_get_end_effector_pose = manipulator.right_get_end_effector_pose:main",
            "right_move_to_pose = manipulator.right_move_to_pose:main",
            "right_arm_driver = manipulator.right_arm_driver:main",
        ],
    },
)
