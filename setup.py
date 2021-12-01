import setuptools
import subprocess


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


# build TCC binary
subprocess.run("mkdir bin".split(" "), cwd='extern/TCC')
subprocess.run("cmake -S . -B build".split(" "), cwd='extern/TCC')
subprocess.run("make install".split(" "), cwd='extern/TCC/build')
subprocess.run("cp extern/TCC/bin/tcc src/tcc/tcc".split(" "))


setuptools.setup(
    name="tcclib",
    version="0.0.1",
    author="Yushi Yang",
    author_email="yangyushi1992@icloud.com",
    description="A Python Wrapper for Topological Cluster Classification",
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yangyushi/tcclib",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},  # location of Distribution package
    packages=setuptools.find_packages(where="src"),  # find import package
    package_data={"": ["tcc"]},
    python_requires=">=3.5",
)
