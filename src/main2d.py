# BSD 2-Clause License
#
# Copyright (c) 2024, Christoph Neuhauser
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
import math
import random
import datetime
import queue
import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import pydens2d

# Bayesian optimization
# conda install -c conda-forge bayesian-optimization
from bayes_opt import BayesianOptimization, UtilityFunction
import pylimbo

# For rasterization
# conda install conda-forge::cairosvg
import io
from PIL import Image
import cairosvg


@njit
def vec_length(p):
    return np.sqrt(p[0] * p[0] + p[1] * p[1])


@njit
def vec_normalize(p):
    l = vec_length(p)
    return np.array([p[0] / l, p[1] / l])


def matrix_translation(t):
    return np.array([
        [1.0, 0.0, t[0]],
        [0.0, 1.0, t[1]],
        [0.0, 0.0, 1.0]
    ])


def sample_view_matrix_circle(aabb):
    theta = 2.0 * np.pi * random.random()
    r_total = 0.5 * vec_length(np.array([aabb[1] - aabb[0], aabb[3] - aabb[2]]))
    r = random.uniform(r_total * 1.25, r_total * 1.75)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    camera_position = np.array([r * -cos_theta, r * -sin_theta])
    return {'cx': camera_position[0], 'cy': camera_position[1], 'theta': theta}


def get_position_random_range(aabb):
    rx = 0.5 * (aabb[1] - aabb[0])
    ry = 0.5 * (aabb[3] - aabb[2])
    r_median = np.median([rx, ry])
    #return [(aabb[i * 2] - 2.0 * r_median, aabb[i * 2 + 1] + 2.0 * r_median) for i in range(2)]
    return [(aabb[i * 2] - 1.0 * r_median, aabb[i * 2 + 1] + 1.0 * r_median) for i in range(2)]


def sample_random_view(aabb):
    theta = random.uniform(0.0, 2.0 * np.pi)
    cam_pos_range = get_position_random_range(aabb)
    camera_position = np.array([np.random.uniform(cam_pos_range[i][0], cam_pos_range[i][1]) for i in range(2)])
    return {'cx': camera_position[0], 'cy': camera_position[1], 'theta': theta}


def check_camera_is_valid(occupation_volume, aabb, cx, cy, fov):
    min_pos = np.array([aabb[0], aabb[2]])
    max_pos = np.array([aabb[1], aabb[3]])

    # Test if the camera does not lie in an occupied voxel.
    occupation_volume_shape = np.array(
        [occupation_volume.shape[1], occupation_volume.shape[0]], dtype=np.int32)
    camera_position = np.array([cx, cy])
    camera_position = (camera_position - min_pos) / (max_pos - min_pos)
    camera_position = camera_position * occupation_volume_shape
    voxel_position = np.empty(2, dtype=np.int32)
    is_outside_volume = False
    outside_dist = 100000
    for i in range(2):
        voxel_position[i] = int(camera_position[i])
        if voxel_position[i] < 0 or voxel_position[i] >= occupation_volume_shape[i]:
            is_outside_volume = True
            if voxel_position[i] < 0:
                outside_dist = min(-voxel_position[i], outside_dist)
            elif voxel_position[i] >= occupation_volume_shape[i]:
                outside_dist = min(voxel_position[i] - occupation_volume_shape[i], outside_dist)
    if not is_outside_volume and occupation_volume[voxel_position[1], voxel_position[0]] != 0:
        return False

    max_outside_dist = 3
    if is_outside_volume and outside_dist < max_outside_dist:
        max_dist = outside_dist + 2
        visited_points = set()
        voxel_queue = queue.Queue()
        voxel_queue.put((0, (voxel_position[0], voxel_position[1])))
        found_neigh = False
        best_neigh_depth = 10000
        # Dist to occupied voxel.
        while not voxel_queue.empty():
            depth, curr_pos = voxel_queue.get()
            for oy in range(-1, 2):
                for ox in range(-1, 2):
                    neigh_pos = (curr_pos[0] + ox, curr_pos[1] + oy)
                    is_neighbor_outside_volume = False
                    for i in range(2):
                        if neigh_pos[i] < 0 or neigh_pos[i] >= occupation_volume_shape[i]:
                            is_neighbor_outside_volume = True
                    if neigh_pos in visited_points:
                        continue
                    if not is_neighbor_outside_volume:
                        if occupation_volume[neigh_pos[1], neigh_pos[0]] != 0:
                            found_neigh = True
                            best_neigh_depth = min(best_neigh_depth, depth + 1)
                    if depth < max_dist:
                        voxel_queue.put((depth + 1, neigh_pos))
                    visited_points.add(neigh_pos)
        if found_neigh and best_neigh_depth <= max_dist:
            return False

    return True


class Circle:
    def __init__(self, cx, cy, r):
        self.cx = cx
        self.cy = cy
        self.r = r

    def contains(self, ptx, pty):
        diffx = ptx - self.cx
        diffy = pty - self.cy
        return np.sqrt(diffx ** 2 + diffy ** 2) <= self.r


def create_density_field_empty_circle(res):
    density_field = np.zeros((res, res), dtype=np.float32)
    sphere0 = Circle(0.0, 0.0, 10.0)
    sphere1 = Circle(0.0, 0.0, 8.0)
    sphere2 = Circle(0.0, 9.0, 1.5)
    for yi in range(res):
        for xi in range(res):
            x = (xi / (res - 1) * 2.0 - 1.0) * 11.0
            y = (yi / (res - 1) * 2.0 - 1.0) * 11.0
            value = 0.0
            if sphere0.contains(x, y) and not sphere1.contains(x, y) and not sphere2.contains(x, y):
                value = 1.0
            density_field[yi, xi] = value
    return density_field


def create_density_field_from_svg(res, svg_path):
    density_field = np.zeros((res, res), dtype=np.float32)
    with open(svg_path, 'rb') as f:
        svg_data = f.read()
    filelike_obj = io.BytesIO(cairosvg.svg2png(svg_data, output_width=res, output_height=res))
    image = Image.open(filelike_obj)
    for yi in range(res):
        for xi in range(res):
            density_field[yi, xi] = 1.0 if image.getpixel((xi, yi))[3] > 0.0 else 0.0
    return density_field


@njit
def ray_box_intersect(b_min, b_max, P, D):
    D[0] = 0.000001 if np.abs(D[0]) <= 0.000001 else D[0]
    D[1] = 0.000001 if np.abs(D[1]) <= 0.000001 else D[1]
    C_Min = (b_min - P) / D
    C_Max = (b_max - P) / D
    tMin = max(min(C_Min[0], C_Max[0]), min(C_Min[1], C_Max[1]))
    tMin = max(0.0, tMin)
    tMax = min(max(C_Min[0], C_Max[0]), max(C_Min[1], C_Max[1]))
    if tMax <= tMin or tMax <= 0:
        return False, None, None
    return True, tMin, tMax


@njit(parallel=True)
def update_visibility_field_py(density_field, visibility_field, aabb, cam_res, cam_pos, theta, fov):
    b_min = np.array([aabb[0], aabb[2]], dtype=np.float32)
    b_max = np.array([aabb[1], aabb[3]], dtype=np.float32)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    front = np.array([cos_theta, sin_theta])
    up = np.array([-sin_theta, cos_theta])
    dist_up = 2.0 * np.tan(fov * 0.5)
    pt0 = front - dist_up * up
    pt1 = front + dist_up * up
    step_size = 0.0001
    for i in range(cam_res):
        t = (i + 0.5) / cam_res
        pt = (1.0 - t) * pt0 + t * pt1
        dir = vec_normalize(pt)
        p = cam_pos
        intersects, t_min, t_max = ray_box_intersect(b_min, b_max, p, dir)
        if intersects:
            p = p + (t_min + 1e-7) * dir
            while True:
                # Check if in AABB:
                is_in_aabb = aabb[0] <= p[0] <= aabb[1] and aabb[2] <= p[1] <= aabb[3]
                if not is_in_aabb:
                    break
                xi = int((p[0] - aabb[0]) / (aabb[1] - aabb[0]) * (res - 1))
                yi = int((p[1] - aabb[2]) / (aabb[3] - aabb[2]) * (res - 1))
                visibility_field[yi, xi] = 1.0
                if density_field[yi, xi] > 0.0:
                    break
                p += step_size * dir


if __name__ == '__main__':
    res = 128
    cam_res = 512
    #density_field = create_density_field_empty_circle(res)
    project_dir = pathlib.Path(__file__).parent.parent.resolve()
    density_field = create_density_field_from_svg(res, f'{project_dir}/data/concave.svg')
    occupation_volume = density_field > 0.0

    aabb = np.array([-1.0, 1.0, -1.0, 1.0], np.float32)
    visibility_field = np.zeros((res, res), dtype=np.float32)

    fov = 0.5
    camera_poses = []
    use_mixed_mode = True
    use_bos = True
    use_python_bos_optimizer = False
    use_visibility_aware_sampling = False
    shall_sample_completely_random_views = False
    #num_sampled_test_views = 128
    num_sampled_test_views = 256
    gains = np.zeros(num_sampled_test_views)

    num_frames = 16
    for i in range(num_frames):
        if use_mixed_mode:
            use_visibility_aware_sampling = i >= num_frames // 2
            shall_sample_completely_random_views = use_visibility_aware_sampling

        if use_visibility_aware_sampling:
            sampled_poses_gain = []
            sampled_poses = []
            if not use_bos:
                tested_poses = []
                for view_idx in range(num_sampled_test_views):
                    is_valid = False
                    while not is_valid:
                        if shall_sample_completely_random_views:
                            pose = sample_random_view(aabb)
                        else:
                            pose = sample_view_matrix_circle(aabb)
                        is_valid = check_camera_is_valid(occupation_volume, aabb, pose['cx'], pose['cx'], fov)
                    cam_pos = np.array([pose['cx'], pose['cy']], dtype=np.float32)
                    transmittance_volume = np.zeros((res, res), dtype=np.float32)
                    pydens2d.update_visibility_field(
                        density_field, transmittance_volume, aabb, cam_res, cam_pos, pose['theta'], fov)
                    gains[view_idx] = (((visibility_field + transmittance_volume).clip(0, 1) - visibility_field) * occupation_volume).sum()
                    tested_poses.append(pose)
                # Get the best view
                idx = gains.argmax().item()
                optimal_gain = gains.max().item()
                best_pose = tested_poses[idx]
                sampled_poses = tested_poses
                sampled_poses_gain = gains
            else:
                def sample_camera_pose_gain_function_params(pose):
                    is_valid = check_camera_is_valid(occupation_volume, aabb, pose['cx'], pose['cx'], fov)
                    if not is_valid:
                        return 0.0
                    cam_pos = np.array([pose['cx'], pose['cy']], dtype=np.float32)
                    transmittance_volume = np.zeros((res, res), dtype=np.float32)
                    pydens2d.update_visibility_field(
                        density_field, transmittance_volume, aabb, cam_res, cam_pos, pose['theta'], fov)
                    gain = (((visibility_field + transmittance_volume).clip(0, 1) - visibility_field) * occupation_volume).sum()
                    sampled_poses.append(pose)
                    sampled_poses_gain.append(gain)
                    return gain
                def sample_camera_pose_gain_function(cx, cy, theta):
                    params = {
                        'cx': cx, 'cy': cy, 'theta': theta
                    }
                    return sample_camera_pose_gain_function_params(params)
                cam_bounds = get_position_random_range(aabb)
                pbounds = {
                    'cx': (cam_bounds[0][0], cam_bounds[0][1]),
                    'cy': (cam_bounds[1][0], cam_bounds[1][1]),
                    'theta': (0.0, 2.0 * np.pi)
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
                            pose = sample_random_view(aabb)
                            is_valid = check_camera_is_valid(occupation_volume, aabb, pose['cx'], pose['cx'], fov)
                        bayesian_optimizer.probe(
                           params=pose, lazy=True,
                        )
                    acquisition_function = UtilityFunction(kind="ucb", kappa=10)
                    bayesian_optimizer.maximize(
                        init_points=0, n_iter=num_sampled_test_views,
                        acquisition_function=acquisition_function
                    )
                    optimal_gain = bayesian_optimizer.max['target']
                    best_pose = bayesian_optimizer.max['params']
                else:
                    init_points = []
                    for view_idx in range(num_sampled_test_views):
                        is_valid = False
                        while not is_valid:
                            pose = sample_random_view(aabb)
                            is_valid = check_camera_is_valid(occupation_volume, aabb, pose['cx'], pose['cx'], fov)
                        init_points.append(pose)
                    settings = pylimbo.BayOptSettings()
                    settings.pbounds = pbounds
                    settings.num_iterations = num_sampled_test_views
                    settings.ucb_kappa = 10
                    optimal_gain, best_pose = pylimbo.maximize(
                        settings, init_points, sample_camera_pose_gain_function_params)

            cam_pos = np.array([best_pose['cx'], best_pose['cy']], dtype=np.float32)
            transmittance_volume = np.zeros((res, res), dtype=np.float32)
            pydens2d.update_visibility_field(
                density_field, transmittance_volume, aabb, cam_res, cam_pos, best_pose['theta'], fov)
            visibility_field = (visibility_field + transmittance_volume).clip(0, 1)
            camera_poses.append(best_pose)

            #plt.plot(range(len(sampled_poses_gain)), sampled_poses_gain)
            #plt.axhline(y=optimal_gain, color='r', linestyle='-')
            #plt.show()
            #camera_poses = camera_poses + sampled_poses
            #break
        else:
            is_valid = False
            while not is_valid:
                if shall_sample_completely_random_views:
                    pose = sample_random_view(aabb)
                else:
                    pose = sample_view_matrix_circle(aabb)
                is_valid = check_camera_is_valid(occupation_volume, aabb, pose['cx'], pose['cx'], fov)
            cam_pos = np.array([pose['cx'], pose['cy']], dtype=np.float32)
            transmittance_volume = np.zeros((res, res), dtype=np.float32)
            pydens2d.update_visibility_field(density_field, transmittance_volume, aabb, cam_res, cam_pos, pose['theta'], fov)
            visibility_field = (visibility_field + transmittance_volume).clip(0, 1)
            camera_poses.append(pose)

    color_field = np.zeros((res, res, 3), dtype=np.ubyte)
    for yi in range(res):
        for xi in range(res):
            color = np.array([255, 255, 255], dtype=np.ubyte)
            v = visibility_field[yi, xi] > 0.0
            d = density_field[yi, xi] > 0.0
            if v and d:
                color[:] = [220, 0, 0]
            elif v and not d:
                color[:] = [255, 200, 200]
            elif not v and d:
                color[:] = [0, 0, 0]
            elif not v and not d:
                color[:] = [255, 255, 255]
            color_field[yi, xi, :] = color[:]

    # Plot cameras & density
    plt.imshow(color_field, interpolation='nearest')
    gray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    dark_red = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    dist = 0.1
    for i, cam_pose in enumerate(camera_poses):
        def conv_pt(p):
            vx = p[0]
            vy = p[1]
            return [(vx - aabb[0]) / (aabb[1] - aabb[0]) * (res - 1), (vy - aabb[2]) / (aabb[3] - aabb[2]) * (res - 1)]
        points = np.zeros((3, 2))
        p = np.array([cam_pose['cx'], cam_pose['cy']])
        cos_theta = np.cos(cam_pose['theta'])
        sin_theta = np.sin(cam_pose['theta'])
        front = np.array([cos_theta, sin_theta])
        up = np.array([-sin_theta, cos_theta])
        dist_up = 2.0 * dist * np.tan(fov * 0.5)
        points[0, :] = conv_pt(p)
        points[1, :] = conv_pt(p + dist * front - dist_up * up)
        points[2, :] = conv_pt(p + dist * front + dist_up * up)
        camera_color = dark_red if use_mixed_mode and i >= num_frames // 2 else gray
        tri = plt.Polygon(points, facecolor="none", edgecolor=camera_color)
        plt.gca().add_patch(tri)
    ax = plt.gca()
    ax.set_xlim([-res, 2 * res])
    ax.set_ylim([-res, 2 * res])
    plt.tight_layout()
    plt.savefig('out.pdf')
    plt.show()
