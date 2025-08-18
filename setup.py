from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    r = f.read().splitlines()

setup(
    name="Custom-gun-object-detection",
    author="Nilesh",
    version="0.1",
    install_requires = r,
    packages=find_packages()
)