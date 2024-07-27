# Version of main.py using Blender instead of vpt for path tracing.
# Due to the use of Blender, this file is released under the GPLv3.
#
# Copyright (C) 2024  Christoph Neuhauser
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import math
import time
import datetime
import argparse
import random
import pathlib
import json
import numpy as np
from vpt import VolumetricPathTracingRenderer
from src.util.sample_view import *

# pip install bpy mathutils
import bpy
import mathutils


def bpy_inspect(obj):
    for prop in dir(obj):
        print("Name: {}, Value: {}, Type:{}".format(prop, getattr(obj, prop), type(getattr(obj, prop))))


def setup_global():
    print(f'Blender {bpy.app.version_string}')

    # Reset scene to be empty.
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Switch to Cycles render engine.
    bpy.data.scenes[0].render.engine = 'CYCLES'

    # Create new world.
    bpy.ops.world.new()
    world = bpy.data.worlds[0]
    bpy.context.scene.world = world
    world.use_nodes = True

    # Set number of frames
    bpy.context.scene.frame_end = num_frames


def prepare_rendering():
    # 3. Set render settings
    bpy.context.scene.render.resolution_x = image_width
    bpy.context.scene.render.resolution_y = image_height
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    # bpy.context.scene.render.image_settings.color_depth = '16'
    # bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    # bpy.context.scene.render.image_settings.color_depth = '32'
    # bpy.context.scene.render.image_settings.use_zbuffer = True  # 32-bit uint
    if transparent_background:
        bpy.context.scene.render.film_transparent = True

    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    #bpy.context.scene.cycles.denoiser = 'OPTIX'
    #bpy.context.scene.cycles.denoising_input_passes = 'RGB'
    bpy.context.scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
    #bpy.context.scene.cycles.denoising_input_passes = 'RGB_ALBEDO'

    # https://blender.stackexchange.com/questions/104651/selecting-gpu-with-python-script
    #bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "OPTIX"  # Alternative: 'NONE', 'CUDA', 'OPTIX', 'HIP', 'ONEAPI'
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = 4
    bpy.context.scene.cycles.volume_bounces = 4
    # https://docs.blender.org/api/2.80/bpy.types.CyclesWorldSettings.html
    #bpy.context.scene.cycles.volume_sampling = 'DISTANCE', 'EQUIANGULAR', 'MULTIPLE_IMPORTANCE'
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(f'Using compute device \'{bpy.context.preferences.addons["cycles"].preferences.compute_device_type}\'')
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1  # Using all devices, include GPU and CPU
        print(f'Using device \'{d["name"]}\'')


def setup_camera():
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 0),
                              rotation=(1.10871, 0.013265, 1.14827), scale=(1, 1, 1))
    camera_obj = bpy.context.object
    camera_obj.location[0] = 0
    # camera_obj.rotation_mode = 'QUATERNION'
    # camera_obj.rotation_quaternion[1] = 1
    # New one is: bpy.ops.wm.obj_import
    # Old one is: bpy.ops.import_scene.obj
    # obj_object = bpy.context.selected_objects[0]
    bpy.context.scene.camera = camera_obj
    camera_obj.data.lens_unit = 'FOV'  # Access it by its object name
    camera_obj.data.angle = fovy  # math.radians(10)
    return camera_obj


def setup_light(use_headlight):
    if use_envmap:
        world = bpy.context.scene.world
        env_image = bpy.data.images.load(
            "/home/christoph/Programming/C++/CloudRendering/Data/CloudDataSets/env_maps/small_empty_room_1_4k_blurred_small.exr")
        node_environment = world.node_tree.nodes.new('ShaderNodeTexEnvironment')
        node_environment.image = env_image
        world.node_tree.links.new(node_environment.outputs["Color"],
                                  bpy.data.worlds["World"].node_tree.nodes["Background"].inputs["Color"])

        world.node_tree.nodes["Background"].inputs[1].default_value = 1.6
        # bg = bpy.data.worlds["World"].node_tree.nodes["Background"]
        # bg.inputs[0].default_value[:3] = (0.5, .1, 0.6)
        # bg.inputs[1].default_value = 1.0
        world.node_tree.nodes["Environment Texture"].interpolation = 'Linear'
        # bpy.data.images["small_empty_room_1_4k_blurred_small.exr"].colorspace_settings.name = 'sRGB'
        # bpy.data.images["small_empty_room_1_4k_blurred_small.exr"].colorspace_settings.name = 'Linear'

    headlight = None
    if use_headlight:
        #bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        headlight = bpy.ops.object.light_add(type='POINT', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    return headlight


def setup_surface():
    triangle_indices = iso_data['triangle_indices'].astype(np.int32)
    vertex_positions = iso_data['vertex_positions']
    vertex_colors = iso_data['vertex_colors']
    vertex_normals = iso_data['vertex_normals']
    num_triangles = triangle_indices.shape[0] // 3
    num_indices = triangle_indices.shape[0]
    num_vertices = vertex_positions.shape[0]
    loop_start = np.arange(start=0, stop=num_indices, step=3, dtype=np.int32)
    loop_total = np.repeat(3, num_triangles).astype(np.int32)
    loop_colors = np.take(vertex_colors, triangle_indices, axis=0)  # [vertex_colors[i] for i in triangle_indices]

    # TODO: Vertex normals

    # For more details see: https://stackoverflow.com/questions/68297161/creating-a-blender-mesh-directly-from-numpy-data
    mesh = bpy.data.meshes.new(name='Isosurface Mesh')
    mesh.vertices.add(num_vertices)
    mesh.vertices.foreach_set("co", vertex_positions.flatten())
    num_vertex_indices = triangle_indices.shape[0]
    mesh.loops.add(num_vertex_indices)
    mesh.loops.foreach_set("vertex_index", triangle_indices)
    mesh.polygons.add(num_triangles)
    mesh.polygons.foreach_set("loop_start", loop_start)
    mesh.polygons.foreach_set("loop_total", loop_total)
    bpy_inspect(mesh)
    bpy_vertex_colors = mesh.vertex_colors.new()
    for i, col in enumerate(bpy_vertex_colors.data):
        col.color[0] = loop_colors[i][0]
        col.color[1] = loop_colors[i][1]
        col.color[2] = loop_colors[i][2]
        col.color[3] = loop_colors[i][3]
    mesh.update()
    mesh.validate()
    obj = bpy.data.objects.new('Isosurface Object', mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.scale[0] = 0.25
    obj.scale[1] = 0.25
    obj.scale[2] = 0.25
    bpy.ops.object.shade_smooth()


def setup_volume():
    bpy.ops.object.volume_add(align='WORLD', location=(0, 0, 0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))
    #bpy.ops.object.volume_import(filepath="/home/christoph/datasets/DisneyCloud/wdas_cloud/wdas_cloud_eighth.vdb", directory="/home/christoph/datasets/DisneyCloud/wdas_cloud/", files=[{"name":"wdas_cloud_eighth.vdb", "name":"wdas_cloud_eighth.vdb"}], relative_path=True, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    volume = bpy.context.object
    volume.data.filepath = os.path.abspath(vdb_path)
    volume.name = "Volume Data"
    # TODO
    volume.scale[0] = 0.005
    volume.scale[1] = 0.005
    volume.scale[2] = 0.005


def setup_cam_poses(camera, pointlight):
    for frame_idx in range(num_frames):
        camera_info = camera_infos[frame_idx]
        location = (camera_info['position'][0], camera_info['position'][1], camera_info['position'][2])
        rot_mat = camera_info['rotation']
        rot_mat_inv = [[rot_mat[j][i] for j in range(0, 3)] for i in range(3)]
        orientation = mathutils.Matrix(rot_mat_inv)
        orientation = orientation.to_quaternion()
        camera.location = location
        camera.rotation_mode = 'QUATERNION'
        camera.rotation_quaternion = orientation
        camera.keyframe_insert(data_path="location", frame=(frame_idx+1))
        camera.keyframe_insert(data_path="rotation_quaternion", frame=(frame_idx+1))
        if pointlight is not None:
            pointlight.location = location
            pointlight.rotation_mode = 'QUATERNION'
            pointlight.rotation_quaternion = orientation
            pointlight.keyframe_insert(data_path="location", frame=(frame_idx + 1))
            pointlight.keyframe_insert(data_path="rotation_quaternion", frame=(frame_idx + 1))


def bpy_render(output_dir, output_file_pattern_string='fg_%d.png'):
    for frame_idx in range(num_frames):
        bpy.context.scene.frame_set(frame_idx + 1)
        bpy.context.scene.render.filepath = os.path.join(output_dir, (output_file_pattern_string % frame_idx))
        bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='vpt_denoise',
        description='Generates volumetric path tracing images.')
    parser.add_argument('-t', '--test_case', default='Brain')
    parser.add_argument('-r', '--img_res', type=int, default=1024)
    parser.add_argument('-n', '--num_frames', type=int, default=2)
    parser.add_argument('-s', '--num_samples', type=int, default=4)
    parser.add_argument('-o', '--out_dir')
    parser.add_argument('--use_const_seed', action='store_true', default=True)  # TODO
    parser.add_argument('--use_headlight', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=True)  # TODO
    args = parser.parse_args()

    test_case = args.test_case
    vpt_renderer = VolumetricPathTracingRenderer()

    if args.use_const_seed:
        random.seed(31)

    num_samples = args.num_samples
    num_frames = args.num_frames
    image_width = args.img_res
    image_height = args.img_res
    aspect = image_width / image_height
    transparent_background = False  # TODO
    use_envmap = not args.use_headlight
    fovy = math.atan(1.0 / 2.0) * 2.0
    shall_sample_completely_random_views = False

    if args.out_dir is None:
        out_dir = f'out_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}'
    else:
        out_dir = args.out_dir
        if out_dir[-1] == '/' or out_dir[-1] == '\\':
            out_dir = out_dir[:-1]
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f'{out_dir}/images').mkdir(exist_ok=True)

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
    elif test_case == 'HollowSphere':
        vpt_renderer.module().load_volume_file(
            str(pathlib.Path.home()) + '/datasets/Toy/vpt/hollowsphere.dat')
    elif test_case == 'Cloud' or test_case == 'Cloud Fog':
        vpt_renderer.module().load_volume_file(
            str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/CloudDataSets/wdas_cloud/wdas_cloud.vdb')
    elif test_case == 'Brain':
        #vpt_renderer.module().load_volume_file(
        #    str(pathlib.Path.home()) + '/datasets/Siemens/brain_cleaned/23.42um_4_cleaned.dat')
        vpt_renderer.module().load_volume_file(
            str(pathlib.Path.home()) + '/datasets/Siemens/brain_cleaned/23.42um_4_cleaned_ds.dat')
    vpt_renderer.module().load_environment_map(
        str(pathlib.Path.home())
        + '/Programming/C++/CloudRendering/Data/CloudDataSets/env_maps/small_empty_room_1_4k_blurred_large.exr')
    vpt_renderer.module().set_use_transfer_function(True)

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
    elif test_case == 'Brain':
        vpt_renderer.module().load_transfer_function_file(
            str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/TransferFunctions/BrainDens.xml')
        vpt_renderer.module().load_transfer_function_file_gradient(
            str(pathlib.Path.home()) + '/Programming/C++/CloudRendering/Data/TransferFunctions/BrainGrad.xml')

    iso_value = 0.0
    use_isosurface = False
    use_volume = True
    if test_case == 'Wholebody':
        use_isosurface = True
        use_gradient_mode = False
        if use_gradient_mode:
            vpt_renderer.module().set_isosurface_type('Gradient')
            iso_value = 0.002
        else:
            vpt_renderer.module().set_isosurface_type('Density')
            iso_value = 0.3
        vpt_renderer.module().set_iso_value(iso_value)
    if test_case == 'Brain':
        use_volume = False
        use_isosurface = True
        vpt_renderer.module().set_use_isosurface_tf(True)
        vpt_renderer.module().set_isosurface_type('Density')
        iso_value = 0.05
        vpt_renderer.module().set_iso_value(iso_value)
    vpt_renderer.module().set_use_isosurfaces(use_isosurface)

    aabb = vpt_renderer.module().get_render_bounding_box()
    rx = 0.5 * (aabb[1] - aabb[0])
    ry = 0.5 * (aabb[3] - aabb[2])
    rz = 0.5 * (aabb[5] - aabb[4])
    radii_sorted = sorted([rx, ry, rz])
    is_spherical = radii_sorted[2] / radii_sorted[0] < 1.9

    camera_infos = []
    for frame_idx in range(num_frames):
        if shall_sample_completely_random_views:
            view_matrix_array, vm, ivm, _ = sample_random_view(aabb)
        elif is_spherical:
            view_matrix_array, vm, ivm = sample_view_matrix_circle(aabb)
        else:
            view_matrix_array, vm, ivm = sample_view_matrix_box(aabb)

        fg_name = f'fg_{frame_idx}.png'
        camera_info = dict()
        camera_info['id'] = frame_idx
        camera_info['fg_name'] = fg_name
        camera_info['width'] = image_width
        camera_info['height'] = image_height
        camera_info['position'] = [ivm[i, 3] for i in range(0, 3)]
        camera_info['rotation'] = [
            [ivm[i, 0] for i in range(0, 3)], [ivm[i, 1] for i in range(0, 3)], [ivm[i, 2] for i in range(0, 3)]
        ]
        camera_info['fovy'] = fovy
        camera_info['aabb'] = aabb
        if test_case != 'HeadDVR':
            camera_info['iso'] = iso_value
        camera_infos.append(camera_info)
        print(f'{frame_idx}/{num_frames}')
    with open(f'{out_dir}/cameras.json', 'w') as f:
        json.dump(camera_infos, f, ensure_ascii=False, indent=4)

    if use_isosurface:
        iso_data = vpt_renderer.triangulate_isosurfaces()
    if use_volume:
        vdb_path = os.path.join(out_dir, 'volume.vdb')
        vpt_renderer.export_vdb_volume(vdb_path)

    # Free the renderer after creating the isosurfaces etc.
    del vpt_renderer

    setup_global()
    camera = setup_camera()
    headlight = setup_light(args.use_headlight)
    if use_isosurface:
        setup_surface()
    if use_volume:
        setup_volume()
    setup_cam_poses(camera, headlight)
    prepare_rendering()
    if args.debug:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, 'test.blend'))

    start = time.time()
    bpy_render(os.path.join(out_dir, 'images'))
    end = time.time()
    print(f'Elapsed time: {end - start}s')

    # Clean up.
    if use_volume and not args.debug:
        os.remove(vdb_path)
