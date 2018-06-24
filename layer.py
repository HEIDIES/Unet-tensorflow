import tensorflow as tf
import ops

def c3s1k64(x, reuse, is_training, norm = None, activation = None, name = 'c3s1k64',
            upsampling = None):
    return ops.conv2d(x, 64, 3, pad_size=1, stride=1, pad_mode='REFLECT', norm=norm,
                      activation=activation, name=name, reuse=reuse, is_training=is_training,
                      upsampling=upsampling)

def c3s1kx2(x, reuse, is_training, norm = None, activation = None, name = 'c3s1kx2',
            upsampling = None):
    return ops.conv2d(x, x.get_shape()[3] * 2, 3, pad_size=1, stride=1, pad_mode='REFLECT', norm=norm,
                      activation=activation, name=name, reuse=reuse, is_training=is_training,
                      upsampling=upsampling)

def c3s1ksame(x, reuse, is_training, norm = None, activation = None, name = 'c3s1ksame',
            upsampling = None):
    return ops.conv2d(x, x.get_shape()[3], 3, pad_size=1, stride=1, pad_mode='REFLECT',
                      norm=norm, activation=activation, name=name, reuse=reuse, is_training=is_training,
                      upsampling=upsampling)

def upc2s1kd2(x, reuse, is_training, norm = None, activation = None, name = 'upc2s1kd2',
            upsampling = [2,2]):
    return ops.conv2d(x, x.get_shape()[3] // 2, 2, padding='SAME', stride=1, norm=norm,
                      activation=activation, name=name, reuse=reuse, is_training=is_training,
                      upsampling=upsampling)

def c3s1kd2(x, reuse, is_training, norm = None, activation = None, name = 'c3s1kd2',
            upsampling = None):
    return ops.conv2d(x, x.get_shape()[3] // 2, 3, pad_size=1, stride=1, pad_mode='REFLECT',
                      norm=norm, activation=activation, name=name, reuse=reuse, is_training=is_training,
                      upsampling=upsampling)

def c3s1k2(x, reuse, is_training, norm = None, activation = None, name = 'c3s1k2',
            upsampling = None):
    return ops.conv2d(x, 2, 3, pad_size=1, stride=1, pad_mode='REFLECT', norm=norm,
                      activation=activation, name=name, reuse=reuse, is_training=is_training,
                      upsampling=upsampling)

def c1s1k1(x, reuse, is_training, norm = None, activation = None, name = 'c1s1k1',
            upsampling = None, kernel_initializer = 'glorot_uniform_sigmoid'):
    return ops.conv2d(x, 1, 1, pad_size=0, stride=1, pad_mode='REFLECT', norm=norm,
                      activation=activation, name=name, reuse=reuse, is_training=is_training,
                      kernel_initializer=kernel_initializer, upsampling=upsampling)