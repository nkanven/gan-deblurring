import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
from scipy import ndimage
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import os

if sys.version_info.major == 3:
    xrange = range


def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)


def ResnetBlock(x, dim, ksize, scope='rb'):
    with tf.variable_scope(scope):
        net = slim.conv2d(x, dim, [ksize, ksize], scope='conv1')
        net = slim.conv2d(net, dim, [ksize, ksize], activation_fn=None, scope='conv2')
        return net + x

def blurImage():
    #face = scipy.misc.face()
    list_dir = os.listdir(r'C:\Users\Nkondog\Desktop\DL_Training_Pix')
    dir_path = r"C:\Users\Nkondog\Desktop\DL_Training_Pix"
    for im in list_dir:
        im = dir_path+"\\"+im
        face = Image.open(im)
        width = 512
        height = 512
        face = face.resize((width, height), Image.BICUBIC)
        #face.show()
        #input()
        #face = face.filter(ImageFilter.GaussianBlur(radius=6))
        face.save(im.split("\\")[-1])
        """with open('datalist_gopro.txt', 'a') as t:
            ti = "GOIMG/sharp/"+im+" GOIMG/blur/"+im+"\n"
            t.write(str(ti))
    #face.show()
    ""print(type(np.asarray(face)))
    input()
    face = np.asarray(face)
    blurred_face = ndimage.gaussian_filter(face, sigma=0)
    very_blurred = ndimage.gaussian_filter(face, sigma=1)
    local_mean = ndimage.uniform_filter(face, size=0)

    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plt.imshow(blurred_face)
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(very_blurred)
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(local_mean)
    plt.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0., top=0.99, bottom=0.01,
                        left=0.01, right=0.99)

    plt.show()"""

#blurImage()