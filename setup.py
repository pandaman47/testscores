from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> list:
    """
    This function reads a requirements file and returns a list of packages.
    """
    requirements = []
    with open(file_path, 'r') as file:
        requirements = file.readlines()
    
    # Remove any whitespace characters like `\n` at the end of each line
    requirements = [req.replace("\n","") for req in requirements]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
name='testscores',
version='0.0.1',
author='Saketh',
author_email='psaketh47@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt'),

description='A package for analyzing and visualizing test scores data',
long_description=open('README.md').read(),

)