from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "mdsv_cpp",
        ["mdsv_bindings.cpp"],
        include_dirs=[
            pybind11.get_include(),
            "/opt/homebrew/Cellar/eigen/3.4.1",  # macOS Homebrew path
            "/opt/homebrew/Cellar/eigen/3.4.1/include/eigen3"
        ],
        extra_compile_args=["-O3", "-std=c++14"],
        language="c++"
    ),
]

setup(
    name="mdsv",
    version="1.0.0",
    author="Your Name",
    description="MDSV with C++ acceleration",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)