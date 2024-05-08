import os
import sys
from setuptools import setup
from setuptools.command.egg_info import egg_info
from pybind11.setup_helpers import Pybind11Extension, build_ext

extra_compile_args = []
if os.name == 'nt':
    extra_compile_args.append('/std:c++17')
    extra_compile_args.append('/openmp')
else:
    extra_compile_args.append('-std=c++17')
    #extra_compile_args.append('-fopenmp=libomp')
    extra_compile_args.append('-fopenmp')


class EggInfoInstallLicense(egg_info):
    def run(self):
        if not self.distribution.have_run.get('install', True):
            self.mkpath(self.egg_info)
            self.copy_file('LICENSE', self.egg_info)
        egg_info.run(self)


setup(
    name='pydens2d',
    author='Christoph Neuhauser',
    ext_modules=[
        Pybind11Extension(
            'pydens2d',
            [
                'src/PyDens2D.cpp',
            ],
            extra_compile_args=extra_compile_args,
        )
    ],
    data_files=[
        ( '.', ['src/pydens2d.pyi'] )
    ],
    cmdclass={
        'build_ext': build_ext,
        'egg_info': EggInfoInstallLicense
    },
    license_files = ('LICENSE',),
    include_dirs=['third_party/eigen']
)
