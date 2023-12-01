# BSD 2-Clause License
#
# Copyright (c) 2022, Christoph Neuhauser
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import numpy as np
from util.module_loader import module_exists, load_module, is_module_loaded


class VolumetricPathTracingRenderer(object):
    """
    Volumetric path tracing renderer module wrapper.
    It uses the C++ extension module from: https://github.com/chrismile/CloudRendering
    Please make sure that the shared library of the module is installed in the directory 'modules/'.
    """

    def __init__(self):
        if not is_module_loaded('vpt'):
            load_module('vpt')
            torch.ops.vpt.initialize()

    def cleanup(self):
        if is_module_loaded('vpt'):
            torch.ops.vpt.cleanup()
            print('cleanup')

    def __del__(self):
        self.cleanup()

    def __call__(self, input_tensor):
        #if importance_sampling_mask is not None:
        #    color_image_relit = torch.ops.sh_aug.render_frame(input_tensor)
        #else:
        #    color_image_relit = torch.ops.sh_aug.render_frame(input_tensor)
        color_image_relit = torch.ops.vpt.render_frame(input_tensor, 16384)
        return color_image_relit

    def module(self):
        return torch.ops.vpt
