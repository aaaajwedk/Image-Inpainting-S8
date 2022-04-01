import logging

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

from neuralgym.ops.layers import resize
from neuralgym.ops.layers import *
from neuralgym.ops.loss_ops import *
from neuralgym.ops.summary_ops import *


logger = logging.getLogger()
np.random.seed(2018)

def bbox2mask(bbox, config, name='mask'):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    def npmask(bbox, height, width, delta_h, delta_w):
        mask = np.zeros((1, height, width, 1), np.float32)
        h = np.random.randint(delta_h//2+1)
        w = np.random.randint(delta_w//2+1)
        mask[:, bbox[0]+h:bbox[0]+bbox[2]-h,
        bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
        return mask
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img_shape = config.IMG_SHAPES
        height = img_shape[0]
        width = img_shape[1]
        mask = tf.py_func(
            npmask,
            [bbox, height, width,
             config.MAX_DELTA_HEIGHT, config.MAX_DELTA_WIDTH],
            tf.float32, stateful=False)
        mask.set_shape([1] + [height, width] + [1])
    return mask



def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    """ Contextual attention layer implementation.

    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.

    Returns:
        tf.Tensor: output

    """
    # get shapes
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    # extract patches from background with stride and rate
    kernel = 2*rate
    raw_w = tf.extract_image_patches(
        b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    f = resize(f, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    b = resize(b, to_shape=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)], func=tf.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        mask = resize(mask, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.extract_image_patches(
        b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.extract_image_patches(
        mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0,1,2], keep_dims=True), 0.), tf.float32)
    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    offsets = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

        # softmax to match
        yi *=  mm  # mask
        yi = tf.nn.softmax(yi*scale, 3)
        yi *=  mm  # mask

        offset = tf.argmax(yi, axis=3, output_type=tf.int32)
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
        y.append(yi)
        offsets.append(offset)
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    return y


def test_contextual_attention(args):
    """Test contextual attention layer with 3-channel image input
    (instead of n-channel feature).

    """
    import cv2
    import os
    # run on cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    rate = 2
    stride = 1
    grid = rate*stride

    b = cv2.imread(args.imageA)
    b = cv2.resize(b, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    h, w, _ = b.shape
    b = b[:h//grid*grid, :w//grid*grid, :]
    b = np.expand_dims(b, 0)
    logger.info('Size of imageA: {}'.format(b.shape))

    f = cv2.imread(args.imageB)
    h, w, _ = f.shape
    f = f[:h//grid*grid, :w//grid*grid, :]
    f = np.expand_dims(f, 0)
    logger.info('Size of imageB: {}'.format(f.shape))

    with tf.Session() as sess:
        bt = tf.constant(b, dtype=tf.float32)
        ft = tf.constant(f, dtype=tf.float32)

        yt, flow = contextual_attention(
            ft, bt, stride=stride, rate=rate,
            training=False, fuse=False)
        y = sess.run(yt)
        cv2.imwrite(args.imageOut, y[0])



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageA', default='', type=str, help='Image A as background patches to reconstruct image B.')
    parser.add_argument('--imageB', default='', type=str, help='Image B is reconstructed with image A.')
    parser.add_argument('--imageOut', default='result.png', type=str, help='Image B is reconstructed with image A.')
    args = parser.parse_args()
    test_contextual_attention(args)
