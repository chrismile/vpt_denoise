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
import sys
import itertools
import pathlib
import subprocess
import html
import smtplib
import ssl
from email.message import EmailMessage
from email.headerregistry import Address
from email.utils import formatdate


def send_mail(
        sender_name, sender_email_address, user_name, password,
        recipient_name, recipient_email_address,
        subject, message_text_raw, message_text_html):
    if sender_email_address.endswith('@in.tum.de'):
        smtp_server = 'mail.in.tum.de'
        port = 587  # STARTTLS
    elif sender_email_address.endswith('@tum.de'):
        smtp_server = 'postout.lrz.de'
        port = 587  # STARTTLS
    else:
        raise Exception(f'Error: Unexpected provider in e-mail address {sender_email_address}!')

    context = ssl.create_default_context()
    server = smtplib.SMTP(smtp_server, port)
    server.ehlo()
    server.starttls(context=context)
    server.ehlo()
    server.login(user_name, password)

    message = EmailMessage()
    message['Subject'] = subject
    message['From'] = Address(display_name=sender_name, addr_spec=sender_email_address)
    message['To'] = Address(display_name=recipient_name, addr_spec=recipient_email_address)
    message['Date'] = formatdate(localtime=True)
    message.set_content(message_text_raw)
    message.add_alternative(message_text_html, subtype='html')

    server.sendmail(sender_email_address, recipient_email_address, message.as_string())

    server.quit()


def escape_html(s):
    s_list = html.escape(s, quote=False).splitlines(True)
    s_list_edit = []
    for se in s_list:
        se_notrail = se.lstrip()
        new_se = se_notrail
        for i in range(len(se) - len(se_notrail)):
            new_se = '&nbsp;' + new_se
        s_list_edit.append(new_se)
    s = ''.join(s_list_edit)
    return s.replace('\n', '<br/>\n')


commands = [
    #[
    #    'python3', 'src/main.py', '--img_res', '2048', '--num_frames', '128',
    #    '--use_headlight', '-o', os.path.join(pathlib.Path.home(), 'datasets/VPT/brain/preset1')
    #],
    #[
    #    'python3', 'src/main.py', '--img_res', '2048', '--num_frames', '128',
    #    '-o', os.path.join(pathlib.Path.home(), 'datasets/VPT/brain/preset2')
    #],
    #[
    #    'python3', os.path.join(pathlib.Path.home(), 'Programming/DL/gaussian_me/run.py')
    #]
    #[
    #    'python3', 'src/main.py', '--img_res', '2048', '--num_frames', '128',
    #    '--envmap',
    #    os.path.join(pathlib.Path.home(), 'Programming/C++/CloudRendering/Data/CloudDataSets/env_maps/constant.exr'),
    #    '-o', os.path.join(pathlib.Path.home(), 'datasets/VPT/brain/preset3')
    #],

    # Cloud data
    #[
    #    'python3', 'src/main.py',
    #    #'python3', 'src/render_blender.py',
    #    '--use_black_bg', '--scale_pos', '0.5',
    #    #'--num_frames', '4',
    #    '--file', os.path.join(pathlib.Path.home(), 'datasets/Han/cloud_simulation_cameras/incomming_0057/incomming_0057.vdb'),
    #    '--camposes', os.path.join(pathlib.Path.home(), 'datasets/Han/cloud_simulation_cameras/incomming_0057/images.json'),
    #    '--img_res', '1024', '--num_samples', '256', '--denoiser', 'Default',
    #    '--scattering_albedo', '0.99', '--extinction_scale', '400.0',
    #    '-o', os.path.join(pathlib.Path.home(), 'datasets/VPT/multiclouds/spp_256_optix/incomming_0057')
    #],
    #[
    #    'python3', 'src/main.py',
    #    # 'python3', 'src/render_blender.py',
    #    '--use_black_bg', '--scale_pos', '0.5',
    #    #'--num_frames', '4',
    #    '--file', os.path.join(pathlib.Path.home(), 'datasets/Han/cloud_simulation_cameras/incomming_0057/incomming_0057.vdb'),
    #    '--camposes', os.path.join(pathlib.Path.home(), 'datasets/Han/cloud_simulation_cameras/incomming_0057/images.json'),
    #    '--img_res', '1024', '--num_samples', '256', '--denoiser', 'None',
    #    '--scattering_albedo', '0.99', '--extinction_scale', '400.0',
    #    '-o', os.path.join(pathlib.Path.home(), 'datasets/VPT/multiclouds/spp_256_noisy/incomming_0057')
    #],
]

for samples in []:  # [4, 256]
    commands.append([
        'python3', 'src/main.py',
        # 'python3', 'src/render_blender.py',
        '--use_black_bg', '--scale_pos', '0.5',
        # '--num_frames', '4',
        '--write_performance_info', '--envmap',
        os.path.join(pathlib.Path.home(), 'Programming/C++/CloudRendering/Data/CloudDataSets/env_maps/environment_han.exr'),
        '--file', os.path.join(pathlib.Path.home(), 'datasets/Han/cloud_simulation_cameras/incomming_0050/incomming_0050.vdb'),
        '--camposes', os.path.join(pathlib.Path.home(), 'datasets/Han/cloud_simulation_cameras/incomming_0050/images.json'),
        '--img_res', '1024', '--num_samples', f'{samples}', '--denoiser', 'None',
        '--scattering_albedo', '0.99', '--extinction_scale', '400.0',
        '-o', os.path.join(pathlib.Path.home(), f'datasets/VPT/multiclouds/spp_{samples}_noisy/incomming_0050')
    ])
    commands.append([
        'python3', 'src/main.py',
        # 'python3', 'src/render_blender.py',
        '--use_black_bg', '--scale_pos', '0.5',
        # '--num_frames', '4',
        '--write_performance_info', '--envmap',
        os.path.join(pathlib.Path.home(), 'Programming/C++/CloudRendering/Data/CloudDataSets/env_maps/environment_han.exr'),
        '--file', os.path.join(pathlib.Path.home(), 'datasets/Han/cloud_simulation_cameras/incomming_0050/incomming_0050.vdb'),
        '--camposes', os.path.join(pathlib.Path.home(), 'datasets/Han/cloud_simulation_cameras/incomming_0050/images.json'),
        '--img_res', '1024', '--num_samples', f'{samples}', '--denoiser', 'Default',
        '--scattering_albedo', '0.99', '--extinction_scale', '400.0',
        '-o', os.path.join(pathlib.Path.home(), f'datasets/VPT/multiclouds/spp_{samples}_optix/incomming_0050')
    ])
    commands.append([
        'python3', 'src/main.py',
        '--use_black_bg', '--scale_pos', '0.5',
        '--write_performance_info', '--envmap',
        os.path.join(pathlib.Path.home(), 'Programming/C++/CloudRendering/Data/CloudDataSets/env_maps/environment_han.exr'),
        '--file', os.path.join(pathlib.Path.home(), 'datasets/Han/cloud_simulation_cameras/incomming_0050/incomming_0050.vdb'),
        '--camposes', os.path.join(pathlib.Path.home(), 'datasets/Han/cloud_simulation_cameras/incomming_0050/images.json'),
        '--img_res', '1024', '--num_samples', f'{samples}', '--denoiser', 'PyTorch Denoiser Module', '--pytorch_denoiser_model_file',
        os.path.join(pathlib.Path.home(), 'Programming/C++/CloudRendering/Data/PyTorch/timm/network_main.pt'),
        '--scattering_albedo', '0.99', '--extinction_scale', '400.0',
        '-o', os.path.join(pathlib.Path.home(), f'datasets/VPT/multiclouds/spp_{samples}_pytorch/incomming_0050')
    ])

#for samples in [1024]:
#    for time_step in range(100, 200):
#    #for time_step in [100, 149, 199]:
#        t = (time_step - 100) / 99.0
#        commands.append([
#            'python3', 'src/main.py',
#            '--use_black_bg', '--scale_pos', '0.5', '--write_performance_info',
#            '--envmap', os.path.join(pathlib.Path.home(), 'Programming/C++/CloudRendering/Data/CloudDataSets/env_maps/environment_han.exr'),
#            '--file', os.path.join(pathlib.Path.home(), f'datasets/Han/flow_super_res/incomming_{time_step:04d}_upsampledQ.vdb'),
#            '--camposes', os.path.join(pathlib.Path.home(), f'datasets/Han/flow_super_res_cameras/incomming_{time_step:04d}_upsampledQ/images.json'),
#            '--num_frames', '16',
#            '--animate_envmap', '2', '--time', str(t),
#            '--img_res', '1024', '--num_samples', f'{samples}', '--denoiser', 'None',
#            '--scattering_albedo', '0.99', '--extinction_scale', '400.0',
#            '-o', os.path.join(pathlib.Path.home(), f'datasets/VPT/multiclouds_upscaled/timeseries_spp_{samples}_noisy/incomming_{time_step:04d}')
#        ])

for samples in [2048]:
    #for time_step in range(50, 200):
    for time_step in [196]:
        t = (time_step - 50) / 149.0
        commands.append([
            'python3', 'src/main.py',
            '--use_black_bg', '--scale_pos', '0.5', '--write_performance_info',
            '--envmap', os.path.join(pathlib.Path.home(), 'Programming/C++/CloudRendering/Data/CloudDataSets/env_maps/belfast_sunset_puresky_4k_2.exr'),
            '--file', os.path.join(pathlib.Path.home(), f'datasets/Han/flow_super_res/incomming_{time_step:04d}_upsampledQ.vdb'),
            '--camposes', os.path.join(pathlib.Path.home(), f'datasets/Han/flow_super_res_cameras/incomming_{time_step:04d}_upsampledQ/images.json'),
            #'--num_frames', '16',
            '--animate_envmap', '3', '--time', str(t),
            '--img_res', '1024', '--num_samples', f'{samples}', '--denoiser', 'None',
            '--scattering_albedo', '0.5', '--extinction_scale', '600.0',
            '-o', os.path.join(pathlib.Path.home(), f'datasets/VPT/multiclouds_upscaled/timeseries_spp_{samples}_noisy/incomming_{time_step:04d}')
        ])

#brain_presets = []
#brain_presets = [1, 2, 3, 4]
brain_presets = []
if 1 in brain_presets:
    commands.append([
        'python3', 'src/main.py', '--test_case', 'Brain', '--img_res', '2048', '--num_frames', '128',
        '--use_headlight', '-o', os.path.join(pathlib.Path.home(), 'datasets/VPT/brain/preset1')
    ])
if 2 in brain_presets:
    commands.append([
        'python3', 'src/main.py', '--test_case', 'Brain', '--img_res', '2048', '--num_frames', '128',
        '-o', os.path.join(pathlib.Path.home(), 'datasets/VPT/brain/preset2')
    ])
if 3 in brain_presets:
    commands.append([
        'python3', 'src/main.py', '--test_case', 'Brain', '--img_res', '2048', '--num_frames', '128',
        '--envmap',
        os.path.join(pathlib.Path.home(), 'Programming/C++/CloudRendering/Data/CloudDataSets/env_maps/constant.exr'),
        '-o', os.path.join(pathlib.Path.home(), 'datasets/VPT/brain/preset3')
    ])
if 4 in brain_presets:
    commands.append([
        'python3', 'src/main.py', '--test_case', 'Brain', '--img_res', '2048', '--num_frames', '128',
        '--render_mode', 'Ray Marching (Emission/Absorption)',
        '-o', os.path.join(pathlib.Path.home(), 'datasets/VPT/brain/preset4')
    ])

# The following code is for training 3DGS models; the script train.py is currently not yet publicly released.
shall_train_3dgs = False
train_script = os.path.join(pathlib.Path.home(), 'Programming/DL/gaussian_me/train.py')
train_3dgs = os.path.exists(train_script)
if shall_train_3dgs and train_3dgs:
    #res = 1
    scenes = ["brain_siemens"]
    presets = [1, 2]
    res = 2
    #scenes = ["brain"]
    #presets = [1, 2, 3, 4]
    #presets = [4]
    settings = list(itertools.product(scenes, presets))
    for (scene, preset) in settings:
        densify_grad_threshold = '0.0001'
        if scene == 'brain_siemens':
            if res == 1:
                densify_grad_threshold = '0.001'
            elif res == 2:
                densify_grad_threshold = '0.0002'
        scene_path = os.path.join(pathlib.Path.home(), "datasets/VPT", scene, f"preset{preset}")
        model_path = os.path.join(pathlib.Path.home(), "datasets/3dgs", f"{scene}_preset{preset}_{res}")
        images_folder = 'images' if res == 1 else f'images_{res}'
        commands.append([
            'python3', train_script,
            '-s', scene_path,
            '-m', model_path,
            '--eval',
            '--test_iterations', '7000', '15000', '30000',
            '--images', images_folder,
            '--densify_grad_threshold', densify_grad_threshold,
            '--save_iterations', '7000', '15000', '30000'
        ])


if __name__ == '__main__':
    shall_send_email = True
    pwd_path = os.path.join(pathlib.Path.home(), 'Documents', 'mailpwd.txt')
    use_email = pathlib.Path(pwd_path).is_file()
    if use_email:
        with open(pwd_path, 'r') as file:
            lines = [line.rstrip() for line in file]
            sender_name = lines[0]
            sender_email_address = lines[1]
            user_name = lines[2]
            password = lines[3]
            recipient_name = lines[4]
            recipient_email_address = lines[5]

    for command in commands:
        print(f"Running '{' '.join(command)}'...")
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = proc.communicate()
        proc_status = proc.wait()
        if proc_status != 0:
            stderr_string = err.decode('utf-8')
            stdout_string = output.decode('utf-8')

            if use_email:
                message_text_raw = f'The following command failed with code {proc_status}:\n'
                message_text_raw += ' '.join(command) + '\n\n'
                message_text_raw += '--- Output from stderr ---\n'
                message_text_raw += stderr_string
                message_text_raw += '---\n\n'
                message_text_raw += '--- Output from stdout ---\n'
                message_text_raw += stdout_string
                message_text_raw += '---'

                message_text_html = \
                    '<html>\n<head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"></head>\n<body>\n'
                message_text_html += f'The following command failed with code {proc_status}:<br/>\n'
                message_text_html += ' '.join(command) + '<br/><br/>\n\n'
                message_text_html += '<font color="red" style="font-family: \'Courier New\', monospace;">\n'
                message_text_html += '--- Output from stderr ---<br/>\n'
                message_text_html += escape_html(stderr_string)
                message_text_html += '---</font>\n<br/><br/>\n\n'
                message_text_html += '<font style="font-family: \'Courier New\', monospace;">\n'
                message_text_html += '--- Output from stdout ---<br/>\n'
                message_text_html += escape_html(stdout_string)
                message_text_html += '---</font>\n'
                message_text_html += '</body>\n</html>'

                if shall_send_email:
                    send_mail(
                        sender_name, sender_email_address, user_name, password,
                        recipient_name, recipient_email_address,
                        'Error while generating images', message_text_raw, message_text_html)

            print('--- Output from stdout ---')
            print(stdout_string.rstrip('\n'))
            print('---\n')
            print('--- Output from stderr ---', file=sys.stderr)
            print(stderr_string.rstrip('\n'), file=sys.stderr)
            print('---', file=sys.stderr)
            sys.exit(1)
            #raise Exception(f'Process returned error code {proc_status}.')
        elif not shall_send_email:
            stderr_string = err.decode('utf-8')
            stdout_string = output.decode('utf-8')
            print('--- Output from stdout ---')
            print(stdout_string.rstrip('\n'))
            print('---\n')
            print('--- Output from stderr ---', file=sys.stderr)
            print(stderr_string.rstrip('\n'), file=sys.stderr)
            print('---', file=sys.stderr)

    message_text_raw = 'run.py finished successfully'
    message_text_html = \
        '<html>\n<head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"></head>\n<body>\n'
    message_text_html += 'run.py finished successfully'
    message_text_html += '</body>\n</html>'
    if shall_send_email:
        send_mail(
            sender_name, sender_email_address, user_name, password,
            recipient_name, recipient_email_address,
            'run.py finished successfully', message_text_raw, message_text_html)
    print('Finished.')
