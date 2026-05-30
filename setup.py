from setuptools import setup, find_packages

NAME = 'VideoLingo'
VERSION = '3.0.0'

with open('requirements.txt', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'docs.*']),
    install_requires=requirements,
    python_requires='>=3.10,<3.11',
)
