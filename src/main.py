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
import datetime
import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rotation
import torch
#conda install -c conda-forge openexr-python
import OpenEXR
import Imath
import array
from vpt import VolumetricPathTracingRenderer
import time
#from netCDF4 import Dataset

# Bayesian optimization
# conda install -c conda-forge bayesian-optimization
from bayes_opt import BayesianOptimization, UtilityFunction


#def save_nc(file_path, data):
#    zs = data.shape[0]
#    ys = data.shape[1]
#    xs = data.shape[2]
#    ncfile = Dataset(file_path, mode='w', format='NETCDF4_CLASSIC')
#    zdim = ncfile.createDimension('z', zs)
#    ydim = ncfile.createDimension('y', ys)
#    xdim = ncfile.createDimension('x', xs)
#    outfield_den = ncfile.createVariable('density', np.float32, ('z', 'y', 'x'))
#    outfield_den[:, :, :] = data[:, :, :]
#    ncfile.close()


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
    #r = random.uniform(r_total / 16.0, r_total / 2.0)
    r = random.uniform(r_total * 1.25, r_total * 1.75)
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


def jitter_direction(camera_forward, jitter_rad):
    v1 = np.array([0.0, 0.0, 1.0])
    v2 = camera_forward
    # For more details see:
    # - https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
    # - https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    q_xyz = np.cross(v1, v2)
    q_w = np.inner(v1, v1) * np.inner(v2, v2) + np.dot(v1, v2)
    transform = Rotation.from_quat(np.array([q_xyz[0], q_xyz[1], q_xyz[2], q_w]))
    phi = np.random.uniform(0.0, 2.0 * np.pi)
    theta = np.random.uniform(0, jitter_rad)
    vec_spherical = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    vec_new = transform.apply(vec_spherical)
    return vec_new


def sample_view_matrix_box(aabb):
    global_up = np.array([0.0, 1.0, 0.0])
    rx = 0.5 * (aabb[1] - aabb[0])
    ry = 0.5 * (aabb[3] - aabb[2])
    rz = 0.5 * (aabb[5] - aabb[4])
    radii_sorted = sorted([rx, ry, rz])
    r_base = np.sqrt(radii_sorted[0] ** 2 + radii_sorted[1] ** 2)
    #if random.randint(0, 1) == 0:
    #    r = random.uniform(1.5 * r_base, r_base * 3.0)
    #else:
    #    r = random.uniform(0.5 * r_base, r_base * 1.5)  # TODO
    r = random.uniform(1.5 * r_base, r_base * 3.0)
    h = 2 * radii_sorted[2]
    hi = 0 if rx >= ry and rx >= rz else (1 if ry >= rz else 2)
    r0i = 0 if hi != 0 else 1
    r1i = 2 if hi != 2 else 1
    area_sphere = 4 * (r**2) * np.pi
    area_cylinder = 2 * np.pi * r * h
    area_total = area_sphere + area_cylinder
    pos_rand = random.random() * area_total
    theta = 2.0 * np.pi * random.random()
    camera_position = np.zeros(3)
    if pos_rand < area_cylinder:
        h_pos = (random.random() - 0.5) * h
        camera_position[r0i] = r * np.cos(theta)
        camera_position[r1i] = r * np.sin(theta)
        camera_forward = vec_normalize(camera_position)
        camera_position[hi] = h_pos
    else:
        phi = np.arccos(1.0 - 2.0 * random.random())
        #camera_position = np.array([r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(phi)])
        camera_position[r0i] = r * np.sin(phi) * np.cos(theta)
        camera_position[r1i] = r * np.sin(phi) * np.sin(theta)
        camera_position[hi] = r * np.cos(phi)
        camera_forward = vec_normalize(camera_position)
        if np.cos(phi) > 0.0:
            camera_position[hi] += h / 2
        else:
            camera_position[hi] -= h / 2

    use_camera_jitter_closeup = True
    if use_camera_jitter_closeup and r / r_base < 1.5:
        jitter_rad = 30.0 / 180.0 * np.pi
        camera_forward = jitter_direction(camera_forward, jitter_rad)

    #camera_position[0] += aabb[0]
    #camera_position[1] += aabb[2]
    #camera_position[2] += aabb[4]
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


def sample_random_view_parametrized(params):
    u = params['u']
    v = params['v']
    w = params['w']
    camera_position = np.array([params['cx'], params['cy'], params['cz']])
    h = Rotation.from_quat(np.array([
        np.sqrt(1-u) * np.sin(2*np.pi*v),
        np.sqrt(1-u) * np.cos(2*np.pi*v),
        np.sqrt(u) * np.sin(2*np.pi*w),
        np.sqrt(u) * np.cos(2*np.pi*w)
    ]))
    rotation_matrix = np.zeros((4, 4))
    rotation_matrix[:3, :3] = h.as_matrix()
    rotation_matrix[3, 3] = 1
    inverse_view_matrix = matrix_translation(camera_position).dot(rotation_matrix)
    view_matrix = np.linalg.inv(inverse_view_matrix)
    view_matrix_array = np.empty(16)
    for i in range(4):
        for j in range(4):
            view_matrix_array[i * 4 + j] = view_matrix[j, i]
    return view_matrix_array, view_matrix, inverse_view_matrix


def get_position_random_range(aabb):
    rx = 0.5 * (aabb[1] - aabb[0])
    ry = 0.5 * (aabb[3] - aabb[2])
    rz = 0.5 * (aabb[5] - aabb[4])
    r_median = np.median([rx, ry, rz])
    return [(aabb[i * 2] - 2.0 * r_median, aabb[i * 2 + 1] + 2.0 * r_median) for i in range(3)]


def sample_random_view(aabb):
    # https://stackoverflow.com/questions/31600717/how-to-generate-a-random-quaternion-quickly
    rvec = np.random.uniform(0.0, 1.0, 3)
    u = rvec[0]
    v = rvec[1]
    w = rvec[2]
    cam_pos_range = get_position_random_range(aabb)
    camera_position = np.array([np.random.uniform(cam_pos_range[i][0], cam_pos_range[i][1]) for i in range(3)])
    params = {
        'cx': camera_position[0], 'cy': camera_position[1], 'cz': camera_position[2],
        'u': u, 'v': v, 'w': w
    }
    view_matrix_array, view_matrix, inverse_view_matrix = sample_random_view_parametrized(params)
    return view_matrix_array, view_matrix, inverse_view_matrix, params


def iceil(a, b):
    return -(a // -b)


def build_projection_matrix(fovy, aspect):
    z_near = 0.01
    z_far = 100.0
    tan_half_fovy = np.tan(fovy / 2.0)
    result = np.zeros((4, 4))
    result[0, 0] = 1.0 / (aspect * tan_half_fovy)
    result[1, 1] = 1.0 / tan_half_fovy
    result[2, 2] = z_far / (z_near - z_far)
    result[3, 2] = -1.0
    result[2, 3] = -(z_far * z_near) / (z_far - z_near)
    return result


class Plane:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def get_normal(self):
        return np.array([self.a, self.b, self.c])

    def get_distance(self, pt):
        return self.a * pt[0] + self.b * pt[1] + self.c * pt[2] + self.d

    def is_outside(self, aabb):
        aabb_center = 0.5 * np.array([aabb[1] + aabb[0], aabb[3] + aabb[2], aabb[5] + aabb[4]])
        extent = 0.5 * np.array([aabb[1] - aabb[0], aabb[3] - aabb[2], aabb[5] - aabb[4]])
        center_dist = self.get_distance(aabb_center)
        max_abs_dist = abs(self.a * extent[0] + self.b * extent[1] + self.c * extent[2])
        return -center_dist > max_abs_dist


def check_aabb_visible_in_view_frustum(vp_matrix, aabb):
    # The underlying idea of the following code comes from
    # http:#www.lighthouse3d.com/tutorials/view-frustum-culling/clip-space-approach-implementation-details/
    frustum_planes = []

    # Near plane
    frustum_planes.append(Plane(
            vp_matrix[3, 0] + vp_matrix[2, 0],
            vp_matrix[3, 1] + vp_matrix[2, 1],
            vp_matrix[3, 2] + vp_matrix[2, 2],
            vp_matrix[3, 3] + vp_matrix[2, 3]))

    # Far plane
    frustum_planes.append(Plane(
            vp_matrix[3, 0] - vp_matrix[2, 0],
            vp_matrix[3, 1] - vp_matrix[2, 1],
            vp_matrix[3, 2] - vp_matrix[2, 2],
            vp_matrix[3, 3] - vp_matrix[2, 3]))

    # Left plane
    frustum_planes.append(Plane(
            vp_matrix[3, 0] + vp_matrix[0, 0],
            vp_matrix[3, 1] + vp_matrix[0, 1],
            vp_matrix[3, 2] + vp_matrix[0, 2],
            vp_matrix[3, 3] + vp_matrix[0, 3]))

    # Right plane
    frustum_planes.append(Plane(
            vp_matrix[3, 0] - vp_matrix[0, 0],
            vp_matrix[3, 1] - vp_matrix[0, 1],
            vp_matrix[3, 2] - vp_matrix[0, 2],
            vp_matrix[3, 3] - vp_matrix[0, 3]))

    # Bottom plane
    frustum_planes.append(Plane(
            vp_matrix[3, 0] + vp_matrix[1, 0],
            vp_matrix[3, 1] + vp_matrix[1, 1],
            vp_matrix[3, 2] + vp_matrix[1, 2],
            vp_matrix[3, 3] + vp_matrix[1, 3]))

    # Top plane
    frustum_planes.append(Plane(
            vp_matrix[3, 0] - vp_matrix[1, 0],
            vp_matrix[3, 1] - vp_matrix[1, 1],
            vp_matrix[3, 2] - vp_matrix[1, 2],
            vp_matrix[3, 3] - vp_matrix[1, 3]))

    # Normalize parameters
    for i in range(6):
        normal_length = np.linalg.norm(frustum_planes[i].get_normal())
        frustum_planes[i].a /= normal_length
        frustum_planes[i].b /= normal_length
        frustum_planes[i].c /= normal_length
        frustum_planes[i].d /= normal_length

    # Not visible if all points on negative side of one plane
    for i in range(6):
        if frustum_planes[i].is_outside(aabb):
            return False
    return True


def check_camera_is_valid(occupation_volume, aabb, view_matrix, inverse_view_matrix, fovy, aspect):
    min_pos = np.array([aabb[0], aabb[2], aabb[4]])
    max_pos = np.array([aabb[1], aabb[3], aabb[5]])

    # Test if the camera does not lie in an occupied voxel.
    occupation_volume_shape = np.array(
        [occupation_volume.shape[2], occupation_volume.shape[1], occupation_volume.shape[0]], dtype=np.int32)
    camera_position = inverse_view_matrix[0:3, 3]
    camera_position = (camera_position - min_pos) / (max_pos - min_pos)
    camera_position = camera_position * occupation_volume_shape
    voxel_position = np.empty(3, dtype=np.int32)
    is_outside_volume = False
    for i in range(3):
        voxel_position[i] = int(camera_position[i])
        if voxel_position[i] < 0 or voxel_position[i] >= occupation_volume_shape[i]:
            is_outside_volume = True
            break
    if not is_outside_volume and occupation_volume[voxel_position[2], voxel_position[1], voxel_position[0]] != 0:
        return False

    # Test if the AABB is visible in the camera view frustum.
    projection_matrix = build_projection_matrix(fovy, aspect)
    vp_matrix = projection_matrix.dot(view_matrix)
    if not check_aabb_visible_in_view_frustum(vp_matrix, aabb):
        return False

    return True


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
    aspect = image_width / image_height

    test_tensor_cpu = torch.ones((4, image_height, image_width), dtype=torch.float32, device=cpu_device)
    test_tensor_cuda = torch.ones((4, image_height, image_width), dtype=torch.float32, device=cuda_device)
    #test_tensor_vulkan = torch.ones(1, dtype=torch.float32, device=vulkan_device)
    #test_tensor_vulkan = test_tensor_cpu.to(vulkan_device)
    print(test_tensor_cpu)
    print(test_tensor_cuda)
    #print(test_tensor_vulkan)
    vpt_renderer = VolumetricPathTracingRenderer()
    render_module = vpt_renderer.module()

    out_dir = f'out_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}'
    pathlib.Path(out_dir).mkdir(exist_ok=True)
    #with open(f'{out_dir}/extrinsics.txt', 'w') as f:
    #    f.write(f'{vpt_renderer.module().get_camera_fovy()}')
    camera_infos = []

    test_case = 'Wholebody'
    #test_case = 'Angiography'
    #test_case = 'HeadDVR'

    shall_sample_completely_random_views = True

    data_dir = '/mnt/data/Flow/Scalar/'
    if not os.path.isdir(data_dir):
        data_dir = '/media/christoph/Elements/Datasets/Scalar/'
    if not os.path.isdir(data_dir):
        data_dir = '/home/christoph/datasets/Flow/Scalar/'
    if test_case == 'Wholebody':
        vpt_renderer.module().load_volume_file(
            data_dir + 'Wholebody [512 512 3172] (CT)/wholebody.dat')
    elif test_case == 'Angiography':
        vpt_renderer.module().load_volume_file(
            data_dir + 'Head [416 512 112] (MRT Angiography)/mrt8_angio.dat')
    elif test_case == 'HeadDVR':
        vpt_renderer.module().load_volume_file(
            data_dir + 'Head [256 256 256] (MR)/Head_256x256x256.dat')
    vpt_renderer.module().load_environment_map(
        str(pathlib.Path.home())
        + '/Programming/C++/CloudRendering/Data/CloudDataSets/env_maps/small_empty_room_1_4k_blurred.exr')
    vpt_renderer.module().set_use_transfer_function(True)

    #mode = 'Delta Tracking'
    mode = 'Next Event Tracking'
    #mode = 'Isosurfaces'
    if test_case == 'Wholebody':
        vpt_renderer.module().load_transfer_function_file(
            str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/TransferFunctions/TF_Wholebody3.xml')
        vpt_renderer.module().load_transfer_function_file_gradient(
            str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/TransferFunctions/TF_WholebodyGrad1.xml')
    elif test_case == 'Angiography':
        vpt_renderer.module().load_transfer_function_file(
            str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/TransferFunctions/HeadAngioDens.xml')
        vpt_renderer.module().load_transfer_function_file_gradient(
            str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/TransferFunctions/HeadAngioGrad.xml')
    elif test_case == 'HeadDVR':
        vpt_renderer.module().load_transfer_function_file(
            str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/TransferFunctions/HeadDVR.xml')
        mode = 'Ray Marching (Emission/Absorption)'

    denoiser_name = 'None'
    if mode != 'Ray Marching (Emission/Absorption)':
        denoiser_name = 'OptiX Denoiser'
    if denoiser_name != 'None':
        vpt_renderer.module().set_denoiser(denoiser_name)

    spp = 256
    if mode == 'Delta Tracking':
        spp = 16384
    elif mode == 'Next Event Tracking':
        spp = 256
    elif mode == 'Isosurfaces':
        spp = 256
    elif mode == 'Ray Marching (Emission/Absorption)':
        spp = 16
    vpt_renderer.set_num_frames(spp)
    #if denoiser_name == 'None':
    #    if mode == 'Delta Tracking':
    #        vpt_renderer.set_num_frames(16384)
    #    elif mode == 'Next Event Tracking':
    #        vpt_renderer.set_num_frames(256)
    #    elif mode == 'Isosurfaces':
    #        vpt_renderer.set_num_frames(256)
    #else:
    #    vpt_renderer.set_num_frames(2)
    #vpt_renderer.module().set_vpt_mode_from_name('Delta Tracking')
    vpt_renderer.module().set_vpt_mode_from_name(mode)

    if test_case == 'Wholebody':
        vpt_renderer.module().set_use_isosurfaces(True)
        use_gradient_mode = False
        if use_gradient_mode:
            vpt_renderer.module().set_isosurface_type('Gradient')
            iso_value = 0.002
        else:
            vpt_renderer.module().set_isosurface_type('Density')
            iso_value = 0.3
        vpt_renderer.module().set_iso_value(iso_value)
    elif test_case == 'Angiography':
        vpt_renderer.module().set_use_isosurfaces(True)
        vpt_renderer.module().set_isosurface_type('Gradient')
        vpt_renderer.module().set_iso_value(0.05)
    elif test_case == 'HeadDVR':
        vpt_renderer.module().set_use_isosurfaces(False)
        vpt_renderer.module().set_extinction_scale(10000.0)

    vpt_renderer.module().set_iso_surface_color([0.4, 0.4, 0.4])
    vpt_renderer.module().set_surface_brdf('Lambertian')
    #vpt_renderer.module().set_surface_brdf('Blinn Phong')
    used_feature_maps = ['Cloud Only', 'Background', 'Depth Blended']
    vpt_renderer.module().set_use_feature_maps(used_feature_maps)
    vpt_renderer.module().set_output_foreground_map(True)

    vpt_renderer.module().set_camera_position([0.0, 0.0, 0.3])
    vpt_renderer.module().set_camera_target([0.0, 0.0, 0.0])

    aabb = vpt_renderer.module().get_render_bounding_box()
    rx = 0.5 * (aabb[1] - aabb[0])
    ry = 0.5 * (aabb[3] - aabb[2])
    rz = 0.5 * (aabb[5] - aabb[4])
    radii_sorted = sorted([rx, ry, rz])
    #is_spherical = radii_sorted[2] - radii_sorted[0] < 0.01
    is_spherical = radii_sorted[2] / radii_sorted[0] < 1.5

    start = time.time()

    use_mixed_mode = True
    use_visibility_aware_sampling = True
    if test_case != 'HeadDVR':
        use_visibility_aware_sampling = True
    ds = 2
    if use_visibility_aware_sampling:
        vpt_renderer.module().set_secondary_volume_downscaling_factor(ds)
    # use_bos = False  # Bayesian optimal sampling
    use_bos = use_visibility_aware_sampling  # Bayesian optimal sampling
    num_sampled_test_views = 128
    volume_voxel_size = vpt_renderer.module().get_volume_voxel_size()
    vis_volume_voxel_size = [iceil(x, ds) for x in volume_voxel_size]
    vis = None
    occupation_volume = None
    gains = None
    if use_visibility_aware_sampling:
        vpt_renderer.set_num_frames(1)
        vis = torch.zeros(
            size=(vis_volume_voxel_size[0], vis_volume_voxel_size[1], vis_volume_voxel_size[2]),
            dtype=torch.float32, device=cuda_device)
        gains = torch.zeros(num_sampled_test_views, device=cpu_device)
        # We need to render one frame before being able to call 'compute_occupation_volume'
        vpt_renderer.set_num_frames(1)
        vpt_renderer.module().set_use_feature_maps(['Transmittance Volume'])
        vpt_test_tensor_cuda = vpt_renderer(test_tensor_cuda)
        vpt_test_tensor_cuda = None
        occupation_volume = vpt_renderer.module().compute_occupation_volume(test_tensor_cuda, ds, 3).cpu().numpy()
        occupation_volume_narrow = vpt_renderer.module().compute_occupation_volume(test_tensor_cuda, ds, 0)
        #vis = (1.0 - occupation_volume_narrow).to(device=cuda_device, dtype=torch.float32)
        #occupation_volume_array = occupation_volume.cpu().numpy().astype(np.float32)
        #save_nc('/home/christoph/datasets/Test/occupation.nc', occupation_volume_array)
    fovy = vpt_renderer.module().get_camera_fovy()

    num_frames = 256
    for i in range(num_frames):
        if use_mixed_mode:
            use_visibility_aware_sampling = i >= num_frames // 2
            shall_sample_completely_random_views = use_visibility_aware_sampling

        if use_visibility_aware_sampling:
            vpt_renderer.set_num_frames(1)
            vpt_renderer.module().set_use_feature_maps(['Transmittance Volume'])

            if not use_bos:
                tested_matrices = []
                for view_idx in range(num_sampled_test_views):
                    is_valid = False
                    while not is_valid:
                        if shall_sample_completely_random_views:
                            view_matrix_array, vm, ivm, _ = sample_random_view(aabb)
                        elif is_spherical:
                            view_matrix_array, vm, ivm = sample_view_matrix_circle(aabb)
                        else:
                            view_matrix_array, vm, ivm = sample_view_matrix_box(aabb)
                        is_valid = check_camera_is_valid(occupation_volume, aabb, vm, ivm, fovy, aspect)
                    vpt_renderer.module().overwrite_camera_view_matrix(view_matrix_array)
                    vpt_test_tensor_cuda = vpt_renderer(test_tensor_cuda)
                    transmittance_volume_tensor = vpt_renderer.module().get_transmittance_volume(test_tensor_cuda)
                    #transmittance_array = transmittance_volume_tensor.cpu().numpy()
                    #save_nc('/home/christoph/datasets/Test/vis.nc', transmittance_array)
                    gains[view_idx] = (((vis + transmittance_volume_tensor).clamp(0, 1) - vis) * occupation_volume_narrow).sum().cpu()
                    tested_matrices.append((view_matrix_array, vm, ivm))
                # Get the best view
                idx = gains.argmax().item()
                view_matrix_array, vm, ivm = tested_matrices[idx]
            else:
                def sample_camera_pose_gain_function(cx, cy, cz, u, v, w):
                    #nonlocal vis
                    params = {
                        'cx': cx, 'cy': cy, 'cz': cz, 'u': u, 'v': v, 'w': w
                    }
                    view_matrix_array, vm, ivm = sample_random_view_parametrized(params)
                    vpt_renderer.module().overwrite_camera_view_matrix(view_matrix_array)
                    is_valid = check_camera_is_valid(occupation_volume, aabb, vm, ivm, fovy, aspect)
                    if not is_valid:
                        return 0.0
                    vpt_test_tensor_cuda = vpt_renderer(test_tensor_cuda)
                    transmittance_volume_tensor = vpt_renderer.module().get_transmittance_volume(test_tensor_cuda)
                    gain = (((vis + transmittance_volume_tensor).clamp(0, 1) - vis) * occupation_volume_narrow).sum().cpu()
                    return gain
                cam_bounds = get_position_random_range(aabb)
                pbounds = {
                    'cx': (cam_bounds[0][0], cam_bounds[0][1]),
                    'cy': (cam_bounds[1][0], cam_bounds[1][1]),
                    'cz': (cam_bounds[2][0], cam_bounds[2][1]),
                    'u': (0.0, 1.0),
                    'v': (0.0, 1.0),
                    'w': (0.0, 1.0),
                }
                bayesian_optimizer = BayesianOptimization(
                    f=sample_camera_pose_gain_function,
                    pbounds=pbounds,
                    # random_state=random_state,  # random_state = np.random.RandomState(17)
                )
                for view_idx in range(num_sampled_test_views):
                    is_valid = False
                    while not is_valid:
                        view_matrix_array, vm, ivm, params = sample_random_view(aabb)
                        is_valid = check_camera_is_valid(occupation_volume, aabb, vm, ivm, fovy, aspect)
                    bayesian_optimizer.probe(
                       params=params, lazy=True,
                    )
                acquisition_function = UtilityFunction(kind="ucb", kappa=10)
                bayesian_optimizer.maximize(
                    init_points=0, n_iter=num_sampled_test_views,
                    acquisition_function=acquisition_function
                )
                optimal_gain = bayesian_optimizer.max['target']
                best_cam_pose = bayesian_optimizer.max['params']
                view_matrix_array, vm, ivm = sample_random_view_parametrized(best_cam_pose)

            vpt_renderer.module().overwrite_camera_view_matrix(view_matrix_array)
            vpt_test_tensor_cuda = vpt_renderer(test_tensor_cuda)
            transmittance_volume_tensor = vpt_renderer.module().get_transmittance_volume(test_tensor_cuda)
            vis = (vis + transmittance_volume_tensor).clamp(0, 1)
            vpt_renderer.set_num_frames(spp)
            vpt_renderer.module().set_use_feature_maps(used_feature_maps)
        else:
            if shall_sample_completely_random_views:
                view_matrix_array, vm, ivm, _ = sample_random_view(aabb)
            elif is_spherical:
                view_matrix_array, vm, ivm = sample_view_matrix_circle(aabb)
            else:
                view_matrix_array, vm, ivm = sample_view_matrix_box(aabb)
            vpt_renderer.module().overwrite_camera_view_matrix(view_matrix_array)
            if use_mixed_mode:
                vpt_renderer.set_num_frames(1)
                vpt_renderer.module().set_use_feature_maps(['Transmittance Volume'])
                vpt_test_tensor_cuda = vpt_renderer(test_tensor_cuda)
                transmittance_volume_tensor = vpt_renderer.module().get_transmittance_volume(test_tensor_cuda)
                vis = (vis + transmittance_volume_tensor).clamp(0, 1)
                vpt_renderer.set_num_frames(spp)
                vpt_renderer.module().set_use_feature_maps(used_feature_maps)

        #torch.cuda.synchronize()

        #img_name = f'img_{i}.exr'
        #vpt_test_tensor_cuda = vpt_renderer(test_tensor_cuda)
        #save_tensor_openexr(f'{out_dir}/{img_name}', vpt_test_tensor_cuda.cpu().numpy())

        #fg_name = f'fg_{i}.exr'
        #image_cloud_only = vpt_renderer.module().get_feature_map_from_string(test_tensor_cuda, 'Cloud Only')
        #save_tensor_openexr(f'{out_dir}/{fg_name}', image_cloud_only.cpu().numpy(), use_alpha=True)

        fg_name = f'fg_{i}.exr'
        vpt_test_tensor_cuda = vpt_renderer(test_tensor_cuda)
        save_tensor_openexr(f'{out_dir}/{fg_name}', vpt_test_tensor_cuda.cpu().numpy(), use_alpha=True)

        bg_name = f'bg_{i}.exr'
        image_background = vpt_renderer.module().get_feature_map_from_string(test_tensor_cuda, 'Background')
        save_tensor_openexr(f'{out_dir}/{bg_name}', image_background.cpu().numpy())
        #save_camera_config(f'{out_dir}/intrinsics_{i}.txt', vpt_renderer.module().get_camera_view_matrix())

        depth_name = f'depth_{i}.exr'
        image_depth = vpt_renderer.module().get_feature_map_from_string(test_tensor_cuda, 'Depth Blended')
        #mask = image_depth[1, :, :] > 1e-5
        #image_depth[0, mask] /= image_depth[1, mask]
        save_tensor_openexr(f'{out_dir}/{depth_name}', image_depth.cpu().numpy())

        #vm = vpt_renderer.module().get_camera_view_matrix()
        camera_info = dict()
        camera_info['id'] = i
        #camera_info['img_name'] = img_name
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
        camera_info['aabb'] = aabb
        if test_case != 'HeadDVR':
            camera_info['iso'] = iso_value
        camera_infos.append(camera_info)
        print(f'{i}/{num_frames}')

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

    with open(f'{out_dir}/cameras.json', 'w') as f:
        json.dump(camera_infos, f, ensure_ascii=False, indent=4)

    del vpt_renderer
