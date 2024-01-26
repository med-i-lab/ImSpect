import numpy as np
import os
from PIL import Image
from pyts.image import GramianAngularField, MarkovTransitionField

def Normalize_img(normal_gasd_arr):
    norm_im = normal_gasd_arr[0, :, :]
    norm_im = ((norm_im + 1) / 2 * 255).astype(np.uint8)
    return norm_im

def spec2img(spec, output_path='img.jpeg', image_size=224):
    """
    Converts a 1D spectrum to a 2D image using MTF, GAFS, and GAMD techniques.

    Parameters:
    spec (array_like): The input 1D spectrum.
    output_path (str): The path where the output image will be saved. Default is 'img.jpeg'.
    image_size (int): The size of the output image. Default is 224.

    Returns:
    None: The function saves the 2D image at the specified output path.
    """
    try:
        spec = np.array(spec)

        # MTF
        mtf = MarkovTransitionField(image_size=image_size)
        X_mtf = mtf.fit_transform(spec.reshape(1, -1))

        # GAFS
        gafs = GramianAngularField(image_size=image_size, method='summation')
        X_gafs = gafs.fit_transform(spec.reshape(1, -1))

        # GAMD
        gafd = GramianAngularField(image_size=image_size, method='difference')
        X_gafd = gafd.fit_transform(spec.reshape(1, -1))

        im0 = Normalize_img(X_gafd)
        im1 = Normalize_img(X_gafs)
        im2 = Normalize_img(X_mtf)

        im = np.concatenate([im0[:, :, np.newaxis], im1[:, :, np.newaxis], im2[:, :, np.newaxis]], axis=2)
        im = Image.fromarray(im)
        im.save(output_path)
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
# spec2img(my_spectrum, output_path='output_image.jpeg', image_size=224)
