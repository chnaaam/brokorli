from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="brokorli",
    version="0.1",
    description="",
    author="brokorli",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False
)