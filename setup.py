import os
import shutil
import subprocess
from pathlib import Path

import setuptools
from setuptools.command.build_py import build_py


ROOT = Path(__file__).parent.resolve()
README = ROOT / "README.md"
PACKAGE_DIR = ROOT / "src" / "tcc"
TCC_SOURCE_DIR = ROOT / "extern" / "TCC"


with README.open("r", encoding="utf-8") as fh:
    long_description = fh.read()


def build_tcc_binary(build_temp):
    build_dir = (ROOT / build_temp / "tcc").resolve()
    install_dir = build_dir / "install"
    cmake = resolve_cmake()

    shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            cmake,
            "-S",
            str(TCC_SOURCE_DIR),
            "-B",
            str(build_dir),
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DCMAKE_C_FLAGS=-fcommon",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        ],
        check=True,
    )
    subprocess.run(
        [cmake, "--build", str(build_dir), "--target", "install"],
        check=True,
    )

    binary = install_dir / "tcc"
    if not binary.is_file():
        raise FileNotFoundError(f"Expected TCC binary was not built: {binary}")
    return binary


def resolve_cmake():
    """
    Avoid Python-entrypoint wrappers such as ~/.local/bin/cmake, which can
        break inside pip's isolated build environment.
    """
    system_path = os.pathsep.join(
        ["/usr/bin", "/usr/local/bin", "/opt/homebrew/bin", "/bin"]
    )
    system_cmake = shutil.which("cmake", path=system_path)
    if system_cmake:
        return system_cmake

    cmake = shutil.which("cmake")
    if cmake:
        return cmake

    raise FileNotFoundError(
        "Could not find a usable 'cmake' executable on PATH"
    )


class BuildPy(build_py):
    def run(self):
        super().run()
        build_cmd = self.get_finalized_command("build")
        binary = build_tcc_binary(build_cmd.build_temp)
        target = Path(self.build_lib) / "tcc" / "tcc"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(binary, target)


setuptools.setup(
    name="tcclib",
    version="0.0.3",
    author="Yushi Yang",
    author_email="yangyushi1992@icloud.com",
    description="A Python Wrapper for Topological Cluster Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yangyushi/tcclib",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},  # location of Distribution package
    packages=setuptools.find_packages(where="src"),  # find import package
    package_data={"tcc": ["tcc"]},
    python_requires=">=3.5",
    cmdclass={"build_py": BuildPy},
)
