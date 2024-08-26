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

import numpy as np
from vpt import VolumetricPathTracingRenderer


if __name__ == '__main__':
    # Computes the global VDB file bounds.
    vpt_renderer = VolumetricPathTracingRenderer()
    render_module = vpt_renderer.module()
    max_val = 1e10
    global_world_bbox = np.array([max_val, -max_val, max_val, -max_val, max_val, -max_val], dtype=np.float64)
    global_index_bbox = np.array([max_val, -max_val, max_val, -max_val, max_val, -max_val], dtype=np.int64)
    for time_step in range(200):
        file_path = f'/home/neuhauser/datasets/Han/flow_super_res/incomming_{time_step:04d}_upsampledQ.vdb'
        vpt_renderer.module().load_volume_file(file_path)
        world_bbox = vpt_renderer.module().get_vdb_world_bounding_box()
        index_bbox = vpt_renderer.module().get_vdb_index_bounding_box()
        voxel_size = vpt_renderer.module().get_vdb_voxel_size()
        for i in range(3):
            global_world_bbox[i * 2] = min(global_world_bbox[i * 2], world_bbox[i * 2])
            global_world_bbox[i * 2 + 1] = max(global_world_bbox[i * 2 + 1], world_bbox[i * 2 + 1])
            global_index_bbox[i * 2] = min(global_index_bbox[i * 2], index_bbox[i * 2])
            global_index_bbox[i * 2 + 1] = max(global_index_bbox[i * 2 + 1], index_bbox[i * 2 + 1])
        print(f'World bbox: {world_bbox}')
        print(f'Index bbox: {index_bbox}')
        print(f'Voxel size: {voxel_size}')
        print('')
    print(f'Global world bbox: {global_world_bbox}')
    print(f'Global index bbox: {global_index_bbox}')
