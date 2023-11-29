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

import matplotlib.pyplot as plt
import numpy as np
import torch
from vpt import VolumetricPathTracingRenderer
import time

if __name__ == '__main__':
    cuda_device_idx = 0
    vulkan_device_idx = 0
    cpu_device = torch.device('cpu')
    cuda_device = torch.device(f'cuda:{cuda_device_idx}' if torch.cuda.is_available() else 'cpu')
    vulkan_device = torch.device(f'vulkan:{vulkan_device_idx}' if torch.is_vulkan_available() else 'cpu')

    #print(torch.__version__)
    #print(torch.is_vulkan_available())
    #print(torch.cuda.is_available())
    #print(cpu_device)
    #print(cuda_device)
    #print(vulkan_device)

    image_width = 256
    image_height = 256

    test_tensor_cpu = torch.ones((4, image_height, image_width), dtype=torch.float32, device=cpu_device)
    test_tensor_cuda = torch.ones((4, image_height, image_width), dtype=torch.float32, device=cuda_device)
    #test_tensor_vulkan = torch.ones(1, dtype=torch.float32, device=vulkan_device)
    #test_tensor_vulkan = test_tensor_cpu.to(vulkan_device)
    print(test_tensor_cpu)
    print(test_tensor_cuda)
    #print(test_tensor_vulkan)
    vpt_renderer = VolumetricPathTracingRenderer()
    render_module = vpt_renderer.module()
    vpt_renderer.module().load_volume_file(
        '/mnt/data/Flow/Scalar/Wholebody [512 512 3172] (CT)/wholebody.dat')
    vpt_renderer.module().load_environment_map(
        '/home/neuhauser/Programming/C++/CloudRendering/Data/CloudDataSets/env_maps/small_empty_room_1_4k_blurred.exr')
    #print(vpt_renderer.module().get_camera_position())
    vpt_renderer.module().set_use_transfer_function(True)
    vpt_renderer.module().load_transfer_function_file(
        '/home/neuhauser/Programming/C++/CloudRendering/Data/TransferFunctions/TF_Wholebody2.xml')
    vpt_renderer.module().set_vpt_mode_from_name('Delta Tracking')
    vpt_renderer.module().set_camera_position([0.0, 0.0, 0.3])
    vpt_test_tensor_cpu = vpt_renderer(test_tensor_cpu)
    vpt_test_tensor_cuda = vpt_renderer(test_tensor_cuda)
    #vpt_test_tensor_vulkan = vpt_renderer(test_tensor_vulkan)
    #print(vpt_test_tensor_cpu)
    print(vpt_test_tensor_cuda)
    #print(vpt_test_tensor_vulkan)

    #plt.imshow(vpt_test_tensor_cpu.permute(1, 2, 0))
    plt.imshow(vpt_test_tensor_cuda.cpu().permute(1, 2, 0))
    plt.show()

    del vpt_renderer
