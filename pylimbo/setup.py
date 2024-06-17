import sys
import subprocess
from pathlib import Path
from packaging.version import Version
from setuptools import setup
from setuptools.command.egg_info import egg_info
from torch.utils.cpp_extension import BuildExtension, CppExtension, IS_WINDOWS, IS_MACOS


def run_program(args):
    return subprocess.Popen(args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.read().decode('utf-8').strip()


extra_compile_args = []
if IS_WINDOWS:
    extra_compile_args.append('/std:c++17')
    extra_compile_args.append('/openmp')
elif IS_MACOS:
    extra_compile_args.append('-std=c++17')
    extra_compile_args.append('-fopenmp=libomp')
else:
    extra_compile_args.append('-std=c++17')
    extra_compile_args.append('-fopenmp')

# USE_NLOPT already defined before including Limbo.
# extra_compile_args.append('-DUSE_NLOPT')

if not IS_WINDOWS:
    pkgconfig_path = run_program('which pkg-config')
    if len(pkgconfig_path) > 0:
        tbb_flags = run_program('pkg-config --libs --cflags tbb')
        if len(tbb_flags) > 0:
            extra_compile_args += tbb_flags.split()
            extra_compile_args.append('-DUSE_TBB')
            tbb_version = run_program('pkg-config --modversion tbb')
            if Version(tbb_version) >= Version('2021.0'):
                extra_compile_args.append('-DUSE_TBB_ONEAPI')
            print('Enabling TBB parallelization support.')
        if len(run_program('pkg-config --cflags tbb')) == 0 and Path('/usr/include/tbb').is_dir():
            extra_compile_args.append('-I/usr/include/tbb')


class EggInfoInstallLicense(egg_info):
    def run(self):
        if not self.distribution.have_run.get('install', True):
            self.mkpath(self.egg_info)
            self.copy_file('LICENSE', self.egg_info)
        egg_info.run(self)

setup(
    name='pylimbo',
    author='Christoph Neuhauser',
    ext_modules=[
        CppExtension(
            'pylimbo',
            [
                'src/PyLimbo.cpp',
            ],
            libraries=['nlopt'],
            extra_compile_args=extra_compile_args,
        )
    ],
    data_files=[
        ( '.', ['src/pylimbo.pyi'] )
    ],
    cmdclass={
        'build_ext': BuildExtension,
        'egg_info': EggInfoInstallLicense
    },
    license_files = ('LICENSE',),
    include_dirs=['third_party/limbo/src', 'third_party/eigen']
)
