import tensorflow as tf

def leaky_relu(x):
    # leaky relu activation function.
    return tf.where(tf.greater(x, 0), x, 0.001 * x)

def _weights(name, shape, mean=0.0, stddev=0.02, initializer = None):
    # weights initializer methods.
    if initializer == 'glorot_normal_tanh':
        fin = shape[2]
        fout = shape[3]
        if len(shape) == 2:
            fin = shape[0]
            fout = shape[1]
        fin = tf.cast(fin, tf.float32)
        fout = tf.cast(fout, tf.float32)
        var = tf.get_variable(
            name, shape, initializer=tf.random_normal_initializer(
                mean=mean, stddev=tf.sqrt(2. / (fin + fout)), dtype=tf.float32
            )
        )
    elif initializer == 'glorot_uniform_tanh':
        fin = shape[2]
        fout = shape[3]
        if len(shape) == 2:
            fin = shape[0]
            fout = shape[1]
        fin = tf.cast(fin, tf.float32)
        fout = tf.cast(fout, tf.float32)
        var = tf.get_variable(
            name, shape, initializer=tf.random_uniform_initializer(
                minval=-tf.sqrt(6. / (fin + fout)), dtype=tf.float32
            )
        )
    elif initializer == 'glorot_normal_sigmoid':
        fin = shape[2]
        fout = shape[3]
        if len(shape) == 2:
            fin = shape[0]
            fout = shape[1]
        fin = tf.cast(fin, tf.float32)
        fout = tf.cast(fout, tf.float32)
        var = tf.get_variable(
            name, shape, initializer=tf.random_normal_initializer(
                mean=mean, stddev=4 * tf.sqrt(2. / (fin + fout)), dtype=tf.float32
            )
        )
    elif initializer == 'glorot_uniform_sigmoid':
        fin = shape[2]
        fout = shape[3]
        if len(shape) == 2:
            fin = shape[0]
            fout = shape[1]
        fin = tf.cast(fin, tf.float32)
        fout = tf.cast(fout, tf.float32)
        var = tf.get_variable(
            name, shape, initializer=tf.random_uniform_initializer(
                minval=-4 * tf.sqrt(6. / (fin + fout)), dtype=tf.float32
            )
        )
    elif initializer == 'he_normal':
        fin = shape[2]
        if len(shape) == 2:
            fin = shape[0]
        fin = tf.cast(fin, tf.float32)
        var = tf.get_variable(
            name, shape, initializer=tf.random_normal_initializer(
                mean=mean, stddev=tf.sqrt(2. / fin), dtype=tf.float32
            )
        )
    elif initializer == 'he_uniform':
        fin = shape[2]
        if len(shape) == 2:
            fin = shape[0]
        fin = tf.cast(fin, tf.float32)
        var = tf.get_variable(
            name, shape, initializer=tf.random_normal_initializer(
                mean=mean, stddev=tf.sqrt(6. / fin), dtype=tf.float32
            )
        )
    else:
        var = tf.get_variable(
            name, shape,
            initializer=tf.random_normal_initializer(
                mean=mean, stddev=stddev, dtype=tf.float32
            )
        )
    return var

def _bias(name, shape, constant = 0.0):
    return tf.get_variable(name, shape, initializer = tf.constant_initializer(constant))

def fully_connected(x, output_dims, use_bias = True, is_training = True, reuse = False,
                    name = None, activation = None, norm = None, weights_initializer = None):
    # fully_connected function.
    with tf.variable_scope(name, reuse = reuse):
        weights = _weights(name = 'weights', shape = [x.get_shape()[1], output_dims], initializer=weights_initializer)
        x = tf.matmul(x, weights)
        if use_bias:
            bias = _bias([output_dims])
            x = tf.add(x, bias)
        if norm is not None:
            x = _norm(x, norm, is_training)
        if activation is not None:
            x = activation(x)
        return x


def conv2d(x, filters, ksize, pad_size = 0, stride = 1, pad_mode = 'CONSTANT', padding = 'VALID',
           norm = None, activation = None, name = 'conv2d', reuse = False, is_training = True,
           kernel_initializer = 'he_uniform', use_bias = False, upsampling = None):
    # convolutional function.
    with tf.variable_scope(name, reuse = reuse):
        if upsampling is not None:
            x = tf.tile(x, multiples=[1, upsampling[0], upsampling[1], 1])
        input_shape = x.get_shape()[3]
        weights = _weights('weights', shape = [ksize, ksize, input_shape, filters], initializer=kernel_initializer)
        if pad_size > 0:
            x = tf.pad(x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode = pad_mode)
        x = tf.nn.conv2d(
            x, weights,
            strides = [1, stride, stride, 1], padding = padding,
            name = name
        )
        if use_bias:
            bias = _bias('bias', [filters])
            x = tf.add(x, bias)
        if norm is not None:
            x = _norm(x, norm, is_training)
        if activation is not None:
            x = activation(x)
        return x

def unconv2d(x, output_dims, ksize, stride = 1, norm = None, activation = None,
             name = None, reuse = False, use_bias = False, is_training = True, kernel_initializer = None):
    # unconvolutional layer.
    input_shape = x.get_shape().as_list()
    with tf.variable_scope(name, reuse = reuse):
        weights = _weights(name = 'weights', shape = [ksize, ksize, output_dims, input_shape[3]], initializer=kernel_initializer)
        x = tf.nn.conv2d_transpose(
            x, weights, [input_shape[0], input_shape[1] * stride,
                        input_shape[2] * stride, output_dims],
            strides = [1, stride, stride, 1], padding = 'SAME'
        )
        if use_bias:
            bias = _bias('bias', [output_dims])
            x = tf.add(x, bias)
        if norm is not None:
            x = _norm(x, norm, is_training)
        if activation is not None:
            x = activation(x)
        return x

def _norm(x, norm, is_training, activation = None):
    if norm == 'batch':
        return _batch_norm(x, is_training, activation = activation)
    if norm == 'instance':
        return _instance_norm(x)

def _batch_norm(x, is_training, activation = None):
    # batch normalization.
    with tf.variable_scope('batch_normalization'):
        x = tf.contrib.layers.batch_norm(x,
                                     decay = 0.9,
                                     scale = True,
                                     updates_collections = None,
                                     is_training = is_training)
        if activation is not None:
            x = activation(x)
        return x

def _instance_norm(x, activation = None):
    # instance normalization.
    with tf.variable_scope('instance_norm'):
        depth = x.get_shape()[3]
        scale = _weights('scale', [depth], mean = 1.0)
        offset = _bias('offset', [depth])
        axis = [1, 2]
        mean ,var = tf.nn.moments(x, axis, keep_dims = True)
        inv = tf.rsqrt(var + 1e-5)
        x = scale * (x - mean) * inv + offset
        if activation is not None:
            x = activation(x)
        return x

def max_pool_with_argmax(x, stride = 2):
    # max pooling function with mask of maximal value.
    with tf.variable_scope('maxpooling'):
        _, mask = tf.nn.max_pool_with_argmax(x, ksize = [1, stride, stride, 1],
                                             strides = [1, stride, stride, 1], padding = 'SAME')
        mask = tf.stop_gradient(mask)
        x = tf.nn.max_pool(x, ksize = [1, stride, stride, 1],
                             strides = [1, stride, stride, 1], padding = 'SAME')
        return x, mask

def unpool(x, mask, stride = 2):
    # un max pooling function.
    with tf.variable_scope('unpooling'):
        ksize = [1, stride, stride, 1]
        input_shape = x.get_shape().as_list()

        output_shape = (input_shape[0], input_shape[1] * ksize[1],
                        input_shape[2] * ksize[2], input_shape[3])

        one_like_mask = tf.ones_like(mask)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype = tf.int64),
                                 shape = [input_shape[0], 1, 1, 1])

        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
        feature_range = tf.range(output_shape[3], dtype = tf.int64)
        f = one_like_mask * feature_range

        updates_size = tf.size(x)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(x, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret

def global_avg_pool(x):
    # global average pooling function.
    with tf.variable_scope('global_avg_pooling'):
        input_shape = x.get_shape().as_list()
        return tf.nn.avg_pool(x, ksize = [1, input_shape[1], input_shape[2], 1],
                              strides = [1, input_shape[1], input_shape[2], 1], padding = 'SAME')
