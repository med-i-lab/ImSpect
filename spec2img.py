import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import pyts.image
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField
from PIL import Image

def Normalize_img(normal_gasd_arr):
    norm_im = normal_gasd_arr[0, :, :]
    norm_im = ( (norm_im+1)/2 * 255).astype(np.uint8)
    return norm_im


def spec2img(spec):
    spec = np.array(spec)
    # MTF
    mtf = MarkovTransitionField(image_size=224)
    X_mtf = mtf.fit_transform(spec)
    #GAFS
    gafs = GramianAngularField(image_size=224,method='summation')
    X_gafs = gafs.fit_transform(spec)
    #GAMD
    gafd = GramianAngularField(image_size=224,method='difference')
    X_gafd = gafd.fit_transform(spec)

    im0 = Normalize_img(X_gafd)
    im1 = Normalize_img(X_gafs)
    im2 = Normalize_img(X_mtf)

    im = np.concatenate([im0[:,:, np.newaxis], im1[:,:, np.newaxis], im2[:,:, np.newaxis]], axis=2)
    # plt.imshow(im)
    im = Image.fromarray(im)
    im.save("img.jpeg")
