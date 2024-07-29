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
import queue
import json
import pathlib
import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import torch
#conda install -c conda-forge openexr-python
import OpenEXR
import Imath
import array
from vpt import VolumetricPathTracingRenderer
import time
#from netCDF4 import Dataset
from .util.sample_view import *

# Bayesian optimization
# conda install -c conda-forge bayesian-optimization
from bayes_opt import BayesianOptimization, UtilityFunction
import pylimbo


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
    # http://www.lighthouse3d.com/tutorials/view-frustum-culling/clip-space-approach-implementation-details/
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
    outside_dist = 100000
    for i in range(3):
        voxel_position[i] = int(camera_position[i])
        if voxel_position[i] < 0 or voxel_position[i] >= occupation_volume_shape[i]:
            is_outside_volume = True
            if voxel_position[i] < 0:
                outside_dist = min(-voxel_position[i], outside_dist)
            elif voxel_position[i] >= occupation_volume_shape[i]:
                outside_dist = min(voxel_position[i] - occupation_volume_shape[i], outside_dist)
    if not is_outside_volume and occupation_volume[voxel_position[2], voxel_position[1], voxel_position[0]] != 0:
        return False

    max_outside_dist = 3
    if is_outside_volume and outside_dist < max_outside_dist:
        max_dist = outside_dist + 2
        visited_points = set()
        voxel_queue = queue.Queue()
        voxel_queue.put((0, (voxel_position[0], voxel_position[1], voxel_position[2])))
        found_neigh = False
        best_neigh_depth = 10000
        # Dist to occupied voxel.
        while not voxel_queue.empty():
            depth, curr_pos = voxel_queue.get()
            for oz in range(-1, 2):
                for oy in range(-1, 2):
                    for ox in range(-1, 2):
                        neigh_pos = (curr_pos[0] + ox, curr_pos[1] + oy, curr_pos[2] + oz)
                        is_neighbor_outside_volume = False
                        for i in range(3):
                            if neigh_pos[i] < 0 or neigh_pos[i] >= occupation_volume_shape[i]:
                                is_neighbor_outside_volume = True
                        if neigh_pos in visited_points:
                            continue
                        if not is_neighbor_outside_volume:
                            if occupation_volume[neigh_pos[2], neigh_pos[1], neigh_pos[0]] != 0:
                                found_neigh = True
                                best_neigh_depth = min(best_neigh_depth, depth + 1)
                        if depth < max_dist:
                            voxel_queue.put((depth + 1, neigh_pos))
                        visited_points.add(neigh_pos)
        if found_neigh and best_neigh_depth <= max_dist:
            return False

    # Test if the AABB is visible in the camera view frustum.
    projection_matrix = build_projection_matrix(fovy, aspect)
    vp_matrix = projection_matrix.dot(view_matrix)
    if not check_aabb_visible_in_view_frustum(vp_matrix, aabb):
        return False

    return True


def run_tests(use_bos=False, num_sampled_test_views=128):
    use_python_bos_optimizer = False
    use_energy_approach = True
    shall_sample_completely_random_views = True
    use_mixed_mode = False
    use_visibility_aware_sampling = True
    if test_case == 'HeadDVR':
        use_visibility_aware_sampling = False
    # use_bos = use_visibility_aware_sampling  # Bayesian optimal sampling
    dont_resample_invalid = True

    vis = None
    gains = None
    if use_visibility_aware_sampling:
        if not use_energy_approach:
            vis = torch.zeros(
                size=(vis_volume_voxel_size[0], vis_volume_voxel_size[1], vis_volume_voxel_size[2]),
                dtype=torch.float32, device=cuda_device)
        gains = torch.zeros(num_sampled_test_views, device=cpu_device)

    num_bins_x = 6
    num_bins_y = 4
    num_bins = num_bins_x * num_bins_y
    gamma = 0.25
    energy_term_list = []
    energy_term_field = torch.zeros(
            size=(vis_volume_voxel_size[0], vis_volume_voxel_size[1], vis_volume_voxel_size[2]),
            dtype=torch.float32, device=cuda_device)
    obs_freq_field = torch.zeros(
            size=(vis_volume_voxel_size[0], vis_volume_voxel_size[1], vis_volume_voxel_size[2]),
            dtype=torch.float32, device=cuda_device)
    angular_obs_freq_field = torch.zeros(
            size=(vis_volume_voxel_size[0], vis_volume_voxel_size[1], vis_volume_voxel_size[2], num_bins),
            dtype=torch.float32, device=cuda_device)

    start = time.time()

    for i in range(num_frames):
        if use_mixed_mode:
            use_visibility_aware_sampling = i >= num_frames // 2
            shall_sample_completely_random_views = use_visibility_aware_sampling

        if use_visibility_aware_sampling:
            vpt_renderer.set_num_frames(1)
            vpt_renderer.module().set_use_feature_maps(feature_maps_vis)

            if not use_bos and dont_resample_invalid:
                gains = []
                tested_matrices = []
                sample_idx = 0
                for view_idx in range(num_sampled_test_views):
                    is_valid = False
                    while not is_valid:
                        if shall_sample_completely_random_views:
                            view_matrix_array, vm, ivm, _ = sample_random_view(aabb)
                        elif is_spherical:
                            view_matrix_array, vm, ivm = sample_view_matrix_circle(aabb)
                        else:
                            view_matrix_array, vm, ivm = sample_view_matrix_box(aabb)
                        sample_idx += 1
                        is_valid = check_camera_is_valid(occupation_volume, aabb, vm, ivm, fovy, aspect)
                    if sample_idx > num_sampled_test_views and len(gains) > 1:
                        break
                    vpt_renderer.module().overwrite_camera_view_matrix(view_matrix_array)
                    vpt_test_tensor_cuda = vpt_renderer(test_tensor_cuda)
                    transmittance_volume_tensor = vpt_renderer.module().get_transmittance_volume(test_tensor_cuda)
                    #transmittance_array = transmittance_volume_tensor.cpu().numpy()
                    #save_nc('/home/christoph/datasets/Test/vis.nc', transmittance_array)
                    if not use_energy_approach:
                        gains.append((((vis + transmittance_volume_tensor).clamp(0, 1) - vis) * occupation_volume_narrow).sum().cpu())
                    else:
                        obs_freq_field_cpy = obs_freq_field.clone().detach()
                        angular_obs_freq_field_cpy = angular_obs_freq_field.clone().detach()
                        vpt_renderer.module().update_observation_frequency_fields(
                            num_bins_x, num_bins_y, transmittance_volume_tensor, obs_freq_field_cpy, angular_obs_freq_field_cpy)
                        vpt_renderer.module().compute_energy(
                            i + 1, num_bins_x, num_bins_y, gamma,
                            obs_freq_field_cpy, angular_obs_freq_field_cpy, occupation_volume_narrow, energy_term_field)
                        gains.append(energy_term_field.sum().cpu())
                    tested_matrices.append((view_matrix_array, vm, ivm))
                # Get the best view
                gains = torch.tensor(np.array(gains), device=cpu_device)
                idx = gains.argmax().item()
                view_matrix_array, vm, ivm = tested_matrices[idx]
            elif not use_bos:
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
                    if not use_energy_approach:
                        gains[view_idx] = (((vis + transmittance_volume_tensor).clamp(0, 1) - vis) * occupation_volume_narrow).sum().cpu()
                    else:
                        obs_freq_field_cpy = obs_freq_field.clone().detach()
                        angular_obs_freq_field_cpy = angular_obs_freq_field.clone().detach()
                        vpt_renderer.module().update_observation_frequency_fields(
                            num_bins_x, num_bins_y, transmittance_volume_tensor, obs_freq_field_cpy, angular_obs_freq_field_cpy)
                        vpt_renderer.module().compute_energy(
                            i + 1, num_bins_x, num_bins_y, gamma,
                            obs_freq_field_cpy, angular_obs_freq_field_cpy, occupation_volume_narrow, energy_term_field)
                        gains[view_idx] = energy_term_field.sum().cpu()
                    tested_matrices.append((view_matrix_array, vm, ivm))
                # Get the best view
                idx = gains.argmax().item()
                view_matrix_array, vm, ivm = tested_matrices[idx]
            else:
                def sample_camera_pose_gain_function_params(params):
                    view_matrix_array, vm, ivm = sample_random_view_parametrized(params)
                    vpt_renderer.module().overwrite_camera_view_matrix(view_matrix_array)
                    is_valid = check_camera_is_valid(occupation_volume, aabb, vm, ivm, fovy, aspect)
                    if not is_valid:
                        return 0.0
                    vpt_test_tensor_cuda = vpt_renderer(test_tensor_cuda)
                    transmittance_volume_tensor = vpt_renderer.module().get_transmittance_volume(test_tensor_cuda)
                    if not use_energy_approach:
                        gain = (((vis + transmittance_volume_tensor).clamp(0, 1) - vis) * occupation_volume_narrow).sum().cpu()
                    else:
                        obs_freq_field_cpy = obs_freq_field.clone().detach()
                        angular_obs_freq_field_cpy = angular_obs_freq_field.clone().detach()
                        vpt_renderer.module().update_observation_frequency_fields(
                            num_bins_x, num_bins_y, transmittance_volume_tensor, obs_freq_field_cpy, angular_obs_freq_field_cpy)
                        vpt_renderer.module().compute_energy(
                            i + 1, num_bins_x, num_bins_y, gamma,
                            obs_freq_field_cpy, angular_obs_freq_field_cpy, occupation_volume_narrow, energy_term_field)
                        gain = energy_term_field.sum().cpu()
                    return gain
                def sample_camera_pose_gain_function(cx, cy, cz, u, v, w):
                    params = {
                        'cx': cx, 'cy': cy, 'cz': cz, 'u': u, 'v': v, 'w': w
                    }
                    return sample_camera_pose_gain_function_params(params)
                cam_bounds = get_position_random_range(aabb)
                pbounds = {
                    'cx': (cam_bounds[0][0], cam_bounds[0][1]),
                    'cy': (cam_bounds[1][0], cam_bounds[1][1]),
                    'cz': (cam_bounds[2][0], cam_bounds[2][1]),
                    'u': (0.0, 1.0),
                    'v': (0.0, 1.0),
                    'w': (0.0, 1.0),
                }

                if use_python_bos_optimizer:
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
                else:
                    init_points = []
                    for view_idx in range(num_sampled_test_views):
                        is_valid = False
                        while not is_valid:
                            view_matrix_array, vm, ivm, params = sample_random_view(aabb)
                            is_valid = check_camera_is_valid(occupation_volume, aabb, vm, ivm, fovy, aspect)
                        init_points.append(params)
                    settings = pylimbo.BayOptSettings()
                    settings.pbounds = pbounds
                    settings.num_iterations = num_sampled_test_views
                    settings.ucb_kappa = 10
                    optimal_gain, best_cam_pose = pylimbo.maximize(
                        settings, init_points, sample_camera_pose_gain_function_params)

                view_matrix_array, vm, ivm = sample_random_view_parametrized(best_cam_pose)

            vpt_renderer.module().overwrite_camera_view_matrix(view_matrix_array)
            vpt_test_tensor_cuda = vpt_renderer(test_tensor_cuda)
            transmittance_volume_tensor = vpt_renderer.module().get_transmittance_volume(test_tensor_cuda)
            if not use_energy_approach:
                vis = (vis + transmittance_volume_tensor).clamp(0, 1)
            else:
                vpt_renderer.module().update_observation_frequency_fields(
                    num_bins_x, num_bins_y, transmittance_volume_tensor, obs_freq_field, angular_obs_freq_field)
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
            # if use_mixed_mode:  # Removed for energy term
            vpt_renderer.set_num_frames(1)
            vpt_renderer.module().set_use_feature_maps(feature_maps_vis)
            vpt_test_tensor_cuda = vpt_renderer(test_tensor_cuda)
            transmittance_volume_tensor = vpt_renderer.module().get_transmittance_volume(test_tensor_cuda)
            if not use_energy_approach:
                vis = (vis + transmittance_volume_tensor).clamp(0, 1)
            else:
                vpt_renderer.module().update_observation_frequency_fields(
                    num_bins_x, num_bins_y, transmittance_volume_tensor, obs_freq_field, angular_obs_freq_field)
            vpt_renderer.set_num_frames(spp)
            vpt_renderer.module().set_use_feature_maps(used_feature_maps)

        #torch.cuda.synchronize()

        if not use_energy_approach:
            vpt_renderer.module().update_observation_frequency_fields(
                num_bins_x, num_bins_y, transmittance_volume_tensor, obs_freq_field, angular_obs_freq_field)
        vpt_renderer.module().compute_energy(
            i + 1, num_bins_x, num_bins_y, gamma,
            obs_freq_field, angular_obs_freq_field, occupation_volume_narrow, energy_term_field)
        energy_term = energy_term_field.sum().cpu()
        energy_term_list.append(energy_term)

        #fg_name = f'fg_{i}.exr'
        #vpt_test_tensor_cuda = vpt_renderer(test_tensor_cuda)
        #save_tensor_openexr(f'{out_dir}/{fg_name}', vpt_test_tensor_cuda.cpu().numpy(), use_alpha=True)

        #bg_name = f'bg_{i}.exr'
        #image_background = vpt_renderer.module().get_feature_map_from_string(test_tensor_cuda, 'Background')
        #save_tensor_openexr(f'{out_dir}/{bg_name}', image_background.cpu().numpy())

        #depth_name = f'depth_{i}.exr'
        #image_depth = vpt_renderer.module().get_feature_map_from_string(test_tensor_cuda, 'Depth Blended')
        #save_tensor_openexr(f'{out_dir}/{depth_name}', image_depth.cpu().numpy())

        camera_info = dict()
        camera_info['id'] = i
        #camera_info['fg_name'] = fg_name
        #camera_info['bg_name'] = bg_name
        camera_info['width'] = image_width
        camera_info['height'] = image_height
        camera_info['position'] = [ivm[i, 3] for i in range(0, 3)]
        camera_info['rotation'] = [
            [ivm[i, 0] for i in range(0, 3)], [ivm[i, 1] for i in range(0, 3)], [ivm[i, 2] for i in range(0, 3)]
        ]
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

    return np.array(energy_term_list), end - start


if __name__ == '__main__':
    random.seed(31)
    pylimbo.seed_random(37)
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
    #test_case = 'HollowSphere'

    data_dir = '/mnt/data/Flow/Scalar/'
    if not os.path.isdir(data_dir):
        data_dir = '/home/christoph/datasets/Scalar/'
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
    elif test_case == 'HollowSphere':
        vpt_renderer.module().load_volume_file(
            str(pathlib.Path.home()) + '/datasets/Toy/vpt/hollowsphere.dat')
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
    elif test_case == 'HollowSphere':
        vpt_renderer.module().load_transfer_function_file(
            str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/TransferFunctions/HollowSphere.xml')

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

    iso_value = 0.0
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
        iso_value = 0.05
        vpt_renderer.module().set_iso_value(iso_value)
    elif test_case == 'HeadDVR':
        vpt_renderer.module().set_use_isosurfaces(False)
        vpt_renderer.module().set_extinction_scale(10000.0)
    elif test_case == 'HollowSphere':
        vpt_renderer.module().set_use_isosurfaces(False)

    vpt_renderer.module().set_iso_surface_color([0.4, 0.4, 0.4])
    vpt_renderer.module().set_surface_brdf('Lambertian')
    #vpt_renderer.module().set_surface_brdf('Blinn Phong')
    used_feature_maps = ['Cloud Only', 'Background', 'Depth Blended']
    feature_maps_vis = ['Transmittance Volume']
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

    ds = 3
    vpt_renderer.module().set_secondary_volume_downscaling_factor(ds)
    volume_voxel_size = vpt_renderer.module().get_volume_voxel_size()
    vis_volume_voxel_size = [iceil(x, ds) for x in volume_voxel_size]
    occupation_volume = None

    # We need to render one frame before being able to call 'compute_occupation_volume'
    vpt_renderer.set_num_frames(1)
    vpt_renderer.module().set_use_feature_maps(feature_maps_vis)
    vpt_test_tensor_cuda = vpt_renderer(test_tensor_cuda)
    vpt_test_tensor_cuda = None
    occupation_volume = vpt_renderer.module().compute_occupation_volume(test_tensor_cuda, ds, 3).cpu().numpy()
    occupation_volume_narrow = vpt_renderer.module().compute_occupation_volume(test_tensor_cuda, ds, 0)
    vpt_renderer.module().set_use_feature_maps(used_feature_maps)
    #occupation_volume_array = occupation_volume.cpu().numpy().astype(np.float32)
    #save_nc('/home/christoph/datasets/Test/occupation.nc', occupation_volume_array)
    fovy = vpt_renderer.module().get_camera_fovy()
    num_frames = 32

    # Setting font type is necessary for VMV (Symposium on Vision, Modeling, and Visualization) submission.
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams.update({'font.family': 'Linux Biolinum O'})
    matplotlib.rcParams.update({'font.size': 17.5})

    test_curve = False
    if test_curve:
        energy_term_list_rnd, time_rnd_next = run_tests(use_bos=False, num_sampled_test_views=64)
        energy_term_list_bos, time_bos_next = run_tests(use_bos=True, num_sampled_test_views=64)
        plt.plot([i for i in range(num_frames)], energy_term_list_rnd, label="Random")
        plt.plot([i for i in range(num_frames)], energy_term_list_bos, label="BOS")
        plt.tight_layout()
        plt.show()
    else:
        n_avg = 8
        #n_avg = 1
        views = [4, 8, 16, 32, 64, 128]
        #views = [4, 128]
        views_np = np.array(views, dtype=int)
        time_rnd = np.zeros(len(views))
        time_bos = np.zeros(len(views))
        energy_list_views_rnd = np.zeros(len(views))
        energy_list_views_bos = np.zeros(len(views))
        for i, test_views in enumerate(views):
            print()
            print(f'Starting {test_views} views...')
            for j in range(n_avg):
                energy_term_list_rnd, time_rnd_next = run_tests(use_bos=False, num_sampled_test_views=test_views)
                energy_term_list_bos, time_bos_next = run_tests(use_bos=True, num_sampled_test_views=test_views)
                time_rnd[i] += time_rnd_next
                time_bos[i] += time_bos_next
                energy_list_views_rnd[i] += energy_term_list_rnd[-1]
                energy_list_views_bos[i] += energy_term_list_bos[-1]
        time_rnd /= len(views)
        time_bos /= len(views)
        energy_list_views_rnd /= len(views)
        energy_list_views_bos /= len(views)

        plt.plot(views_np, energy_list_views_rnd, label="Random")
        plt.plot(views_np, energy_list_views_bos, label="BOS")
        plt.xlabel('#Candidate views')
        plt.ylabel('Energy')
        plt.xscale('log')
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{int(y)}'))
        plt.minorticks_off()
        plt.xticks(views_np)
        #plt.ticklabel_format(axis='x', style='plain')
        #print(f'Times Rnd (s): {time_rnd}')
        #print(f'Times BOS (s): {time_bos}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('maximum.pdf', bbox_inches='tight', pad_inches=0.01)
        plt.show()

        plt.plot(views_np, time_rnd, label="Random")
        plt.plot(views_np, time_bos, label="BOS")
        plt.xlabel('#Candidate views')
        plt.ylabel('Time (s)')
        plt.xscale('log')
        plt.yscale('log')
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{int(y)}'))
        plt.minorticks_off()
        plt.xticks(views_np)
        plt.yticks([1, 10, 100])
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('time.pdf', bbox_inches='tight', pad_inches=0.01)
        plt.show()

    del vpt_renderer
