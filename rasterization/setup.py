import sys

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from setuptools import find_namespace_packages

setup(
    name="nbodyhpc-rasterizer",
    version="0.0.1",
    description="A fast rasterizer for estimating fields from point clouds",
    author="Wenda Zhou",
    license="MIT",
    packages=find_namespace_packages(where="src/python", include=["nbodyhpc.*"]),
    package_dir={"": "src/python"},
    cmake_install_dir="src/python/nbodyhpc/rasterizer",
    include_package_data=True,
    extras_require={"test": ["pytest"]},
    python_requires=">=3.6",
)
