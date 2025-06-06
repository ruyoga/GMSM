from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "mdsv.mdsv_cpp",
        ["mdsv/cpp/mdsv_cpp.cpp", "mdsv/cpp/pybind_wrapper.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "mdsv/cpp"
        ],
        cxx_std=11,
        define_macros=[("EIGEN_NO_DEBUG", "1"), ("NDEBUG", "1")],
    ),
]

setup(
    name="mdsv",
    version="0.1.0",
    author="MDSV Python Contributors",
    author_email="",
    description="Multifractal Discrete Stochastic Volatility Model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "statsmodels>=0.12.0",
        "arch>=4.15",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "black", "flake8"],
    },
)