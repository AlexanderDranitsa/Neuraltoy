from setuptools import setup, find_packages

setup(
    name='Neuraltoy',
    version='1.0.0',
    author='Angry Cat',
    packages=find_packages(exclude=['tests']),
    description='Python, recognizing of simple geometric figures',
)
