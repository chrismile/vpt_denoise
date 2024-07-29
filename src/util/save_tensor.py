import numpy as np
from PIL import Image
# conda install -c conda-forge openexr-python
import OpenEXR
import Imath


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


def save_tensor_png(file_path, data):
    # Convert linear RGB to sRGB.
    for i in range(3):
        data[i, :, :] = np.power(data[i, :, :], 1.0 / 2.2)
    data = np.clip(data, 0.0, 1.0)
    data = data.transpose(1, 2, 0)
    data = (data * 255).astype('uint8')
    image_out = Image.fromarray(data)
    image_out.save(file_path)


def convert_image_black_background(input_filename):
    input_image = Image.open(input_filename)
    input_image.putalpha(255)
    input_image.save(input_filename)
