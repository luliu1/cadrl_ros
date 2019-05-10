from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup


setup_args = generate_distutils_setup(
    packages=[
        'crowd_nav',
	'crowd_nav_policy',
        'crowd_sim'
        'crowd_sim_utils',
    ],
    package_dir={'':'src'})

setup(**setup_args)
