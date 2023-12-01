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

import os
import random
import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import torch
#conda install -c conda-forge openexr-python
import OpenEXR
import Imath
import array
from vpt import VolumetricPathTracingRenderer
import time


def save_tensor_openexr(file_path, data, dtype=np.float16, use_alpha=False):
    if dtype == np.float32:
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
    elif dtype == np.float16:
        pt = Imath.PixelType(Imath.PixelType.HALF)
    else:
        raise Exception('Error in save_tensor_openexr: Invalid format.')
    if data.dtype != dtype:
        data = data.astype(dtype)
    header = OpenEXR.Header(data.shape[2], data.shape[1])
    if use_alpha:
        header['channels'] = {
            'R': Imath.Channel(pt), 'G': Imath.Channel(pt), 'B': Imath.Channel(pt), 'A': Imath.Channel(pt)
        }
    else:
        header['channels'] = {'R': Imath.Channel(pt), 'G': Imath.Channel(pt), 'B': Imath.Channel(pt)}
    out = OpenEXR.OutputFile(file_path, header)
    reds = data[0, :, :].tobytes()
    greens = data[1, :, :].tobytes()
    blues = data[2, :, :].tobytes()
    if use_alpha:
        alphas = data[3, :, :].tobytes()
        out.writePixels({'R': reds, 'G': greens, 'B': blues, 'A': alphas})
    else:
        out.writePixels({'R': reds, 'G': greens, 'B': blues})


def save_camera_config(file_path, view_matrix):
    with open(file_path, 'w') as f:
        for i in range(16):
            if i != 0:
                f.write(' ')
            f.write(f'{view_matrix[i]}')


def vec_cross(v0, v1):
    return np.array([
        v0[1] * v1[2] - v0[2] * v1[1],
        v0[2] * v1[0] - v0[0] * v1[2],
        v0[0] * v1[1] - v0[1] * v1[0],
    ])


def vec_length(p):
    return np.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])


def vec_normalize(p):
    l = vec_length(p)
    return np.array([p[0] / l, p[1] / l, p[2] / l])


def matrix_translation(t):
    return np.array([
        [1.0, 0.0, 0.0, t[0]],
        [0.0, 1.0, 0.0, t[1]],
        [0.0, 0.0, 1.0, t[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])


def sample_view_matrix_circle(aabb):
    global_up = np.array([0.0, 1.0, 0.0])
    theta = 2.0 * np.pi * random.random()
    phi = np.arccos(1.0 - 2.0 * random.random())
    r_total = 0.5 * vec_length(np.array([aabb[1] - aabb[0], aabb[3] - aabb[2], aabb[5] - aabb[4]]))
    r = random.uniform(r_total / 16.0, r_total / 2.0)
    camera_position = np.array([r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(phi)])
    camera_forward = vec_normalize(camera_position)
    camera_right = vec_normalize(vec_cross(global_up, camera_forward))
    camera_up = vec_normalize(vec_cross(camera_forward, camera_right))
    rotation_matrix = np.empty((4, 4))
    for i in range(4):
        rotation_matrix[i, 0] = camera_right[i] if i < 3 else 0.0
        rotation_matrix[i, 1] = camera_up[i] if i < 3 else 0.0
        rotation_matrix[i, 2] = camera_forward[i] if i < 3 else 0.0
        rotation_matrix[i, 3] = 0.0 if i < 3 else 1.0
    inverse_view_matrix = matrix_translation(camera_position).dot(rotation_matrix)
    view_matrix = np.linalg.inv(inverse_view_matrix)
    view_matrix_array = np.empty(16)
    for i in range(4):
        for j in range(4):
            view_matrix_array[i * 4 + j] = view_matrix[j, i]
    return view_matrix_array, view_matrix, inverse_view_matrix


def sample_view_matrix_box(aabb):
    global_up = np.array([0.0, 1.0, 0.0])
    theta = 2.0 * np.pi * random.random()
    phi = np.arccos(1.0 - 2.0 * random.random())
    r_total = 0.5 * vec_length(np.array([aabb[1] - aabb[0], aabb[3] - aabb[2], aabb[5] - aabb[4]]))
    r = random.uniform(r_total / 16.0, r_total / 2.0)
    camera_position = np.array([r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(phi)])
    camera_forward = vec_normalize(camera_position)
    camera_right = vec_normalize(vec_cross(global_up, camera_forward))
    camera_up = vec_normalize(vec_cross(camera_forward, camera_right))
    rotation_matrix = np.empty((4, 4))
    for i in range(4):
        rotation_matrix[i, 0] = camera_right[i] if i < 3 else 0.0
        rotation_matrix[i, 1] = camera_up[i] if i < 3 else 0.0
        rotation_matrix[i, 2] = camera_forward[i] if i < 3 else 0.0
        rotation_matrix[i, 3] = 0.0 if i < 3 else 1.0
    inverse_view_matrix = matrix_translation(camera_position).dot(rotation_matrix)
    view_matrix = np.linalg.inv(inverse_view_matrix)
    view_matrix_array = np.empty(16)
    for i in range(4):
        for j in range(4):
            view_matrix_array[i * 4 + j] = view_matrix[j, i]
    return view_matrix_array, view_matrix, inverse_view_matrix


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

    test_mode = False
    image_width = 1024
    image_height = 1024

    test_tensor_cpu = torch.ones((4, image_height, image_width), dtype=torch.float32, device=cpu_device)
    test_tensor_cuda = torch.ones((4, image_height, image_width), dtype=torch.float32, device=cuda_device)
    #test_tensor_vulkan = torch.ones(1, dtype=torch.float32, device=vulkan_device)
    #test_tensor_vulkan = test_tensor_cpu.to(vulkan_device)
    print(test_tensor_cpu)
    print(test_tensor_cuda)
    #print(test_tensor_vulkan)
    vpt_renderer = VolumetricPathTracingRenderer()
    render_module = vpt_renderer.module()

    pathlib.Path('out').mkdir(exist_ok=True)
    #with open('out/extrinsics.txt', 'w') as f:
    #    f.write(f'{vpt_renderer.module().get_camera_fovy()}')
    aabb = vpt_renderer.module().get_render_bounding_box()
    camera_infos = []

    data_dir = '/mnt/data/Flow/Scalar/'
    if not os.path.isdir(data_dir):
        data_dir = '/media/christoph/Elements/Datasets/Scalar/'
    vpt_renderer.module().load_volume_file(
        data_dir + 'Wholebody [512 512 3172] (CT)/wholebody.dat')
    vpt_renderer.module().load_environment_map(
        str(pathlib.Path.home())
        + '/Programming/C++/CloudRendering/Data/CloudDataSets/env_maps/small_empty_room_1_4k_blurred.exr')
    vpt_renderer.module().set_use_transfer_function(True)
    vpt_renderer.module().load_transfer_function_file(
        str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/TransferFunctions/TF_Wholebody3.xml')
    vpt_renderer.module().set_vpt_mode_from_name('Delta Tracking')
    vpt_renderer.module().set_use_isosurfaces(True)
    #vpt_renderer.module().set_iso_value(0.360)
    vpt_renderer.module().set_iso_value(0.3)
    vpt_renderer.module().set_surface_brdf('Lambertian')
    #vpt_renderer.module().set_surface_brdf('Blinn Phong')
    vpt_renderer.module().set_use_feature_maps(["Cloud Only", "Background"])

    vpt_renderer.module().set_camera_position([0.0, 0.0, 0.3])
    vpt_renderer.module().set_camera_target([0.0, 0.0, 0.0])

    start = time.time()

    for i in range(2):
        #view_matrix = sample_view_matrix_circle(aabb)
        view_matrix_array, vm, ivm = sample_view_matrix_box(aabb)
        vpt_renderer.module().overwrite_camera_view_matrix(view_matrix_array)
        #torch.cuda.synchronize()

        img_name = f'img_{i}.exr'
        vpt_test_tensor_cuda = vpt_renderer(test_tensor_cuda)
        save_tensor_openexr(f'out/{img_name}', vpt_test_tensor_cuda.cpu().numpy())

        fg_name = f'fg_{i}.exr'
        image_cloud_only = vpt_renderer.module().get_feature_map_from_string(test_tensor_cuda, "Cloud Only")
        save_tensor_openexr(f'out/{fg_name}', image_cloud_only.cpu().numpy(), use_alpha=True)

        bg_name = f'bg_{i}.exr'
        image_background = vpt_renderer.module().get_feature_map_from_string(test_tensor_cuda, "Background")
        save_tensor_openexr(f'out/{bg_name}', image_background.cpu().numpy())
        #save_camera_config(f'out/intrinsics_{i}.txt', vpt_renderer.module().get_camera_view_matrix())

        #vm = vpt_renderer.module().get_camera_view_matrix()
        camera_info = dict()
        camera_info['id'] = i
        camera_info['img_name'] = img_name
        camera_info['fg_name'] = fg_name
        camera_info['bg_name'] = bg_name
        camera_info['width'] = image_width
        camera_info['height'] = image_height
        camera_info['position'] = [ivm[i, 3] for i in range(0, 3)]
        camera_info['rotation'] = [
            [ivm[i, 0] for i in range(0, 3)], [ivm[i, 1] for i in range(0, 3)], [ivm[i, 2] for i in range(0, 3)]
        ]
        #camera_info['view_matrix'] = [
        #    [vm[i] for i in range(0, 4)], [vm[i] for i in range(4, 8)],
        #    [vm[i] for i in range(8, 12)], [vm[i] for i in range(12, 16)]
        #]
        camera_info['fovy'] = vpt_renderer.module().get_camera_fovy()
        camera_infos.append(camera_info)

        if test_mode:
            #vpt_test_tensor_cpu = vpt_renderer(test_tensor_cpu)
            #vpt_test_tensor_vulkan = vpt_renderer(test_tensor_vulkan)
            # print(vpt_test_tensor_cpu)
            print(vpt_test_tensor_cuda)
            # print(vpt_test_tensor_vulkan)
            # plt.imshow(vpt_test_tensor_cpu.permute(1, 2, 0))
            plt.imshow(vpt_test_tensor_cuda.cpu().permute(1, 2, 0))
            plt.show()
            break

    end = time.time()
    print(f'Elapsed time: {end - start}s')

    with open('out/cameras.json', 'w') as f:
        json.dump(camera_infos, f, ensure_ascii=False, indent=4)

    del vpt_renderer
