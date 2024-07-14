# vpt_denoise

Code highlighting how to use the application [CloudRenderer](https://github.com/chrismile/CloudRendering) to generate
volumetric path tracing images in Python. This can be used for implementing deep learning image denoisers, e.g.,
using the Python interface of PyTorch.

For more details, please refer to: https://github.com/chrismile/CloudRendering#pytorch-module

Please use `-DCMAKE_INSTALL_PREFIX=<path>` or the `build.sh` script (see below) to install the Python module in the top
level directory of this repository.

### Installation instructions

(1) Install Miniconda (a Python package manager): https://docs.conda.io/projects/miniconda/en/latest/

(2) Create a new environment with conda.

(3) Install PyTorch (https://pytorch.org/get-started/locally/), e.g.:
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

(4) Install the corresponding version of the CUDA SDK (e.g., 12.1 for the current version of PyTorch).

NOTE: If the C++ compiler of your distribution does not support the CUDA version of the latest pre-compiled PyTorch version,
please try building PyTorch manually from source (https://github.com/pytorch/pytorch#from-source).

(5) Clone (and optionally fork) the code in this repo: https://github.com/chrismile/vpt_denoise

(6a) Linux: Call `build.sh` as follows:
```shell
./build.sh --use-pytorch --install-dir /path/to/vpt_denoise
```

(6b)  Windows: Edit the `build.bat` script, and add to the cmake arguments when the CloudRenderer build is configured
(please adapt the paths to where conda/the Python code lie):
```
-DCMAKE_PREFIX_PATH=~/miniconda3/envs/vpt/lib/python3.10/site-packages/torch/share/cmake -DBUILD_PYTORCH_MODULE=On -DSUPPORT_PYTORCH_DENOISER=On -DBUILD_KPN_MODULE=On -DCMAKE_INSTALL_PREFIX=~/Programming/DL/vpt_denoise
```
Additionally, `cmake --build .build --target install` needs to be added to install the module after building the application.
