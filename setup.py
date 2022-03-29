from setuptools import setup, find_packages

requirements = []
with open("requirements.txt") as fp:
    for requirement in fp.readlines():
        requirements.append(requirement)

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