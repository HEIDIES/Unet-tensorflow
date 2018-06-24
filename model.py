import tensorflow as tf
import layer
import utils


class Unet:
    def __init__(self, name,  image_size=256, norm='batch', learning_rate=0.01):
        self.name = name
        self.norm = norm
        self.image_size = image_size
        self.learning_rate = learning_rate

        self.x = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, 1],
                                name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, 1],
                                name='y')
        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
        self.reuse = len([var for var in tf.global_variables() if
                          var.name.startswith(self.name)]) > 0
        with tf.variable_scope(self.name):
            conv1 = layer.c3s1k64(self.x, reuse=self.reuse, is_training=self.is_training,
                                  norm=self.norm, activation=tf.nn.relu, name='conv1_1')
            conv1 = layer.c3s1ksame(conv1, reuse=self.reuse, is_training=self.is_training,
                                    norm=self.norm, activation=tf.nn.relu, name='conv1_2')
            pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1],
                                   padding='SAME', name='pool1')
            conv2 = layer.c3s1kx2(pool1, reuse=self.reuse, is_training=self.is_training,
                                  norm=self.norm, activation=tf.nn.relu, name='conv2_1')
            conv2 = layer.c3s1ksame(conv2, reuse=self.reuse, is_training=self.is_training,
                                    norm=self.norm, activation=tf.nn.relu, name='conv2_2')
            pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1],
                                   padding='SAME', name='pool2')
            conv3 = layer.c3s1kx2(pool2, reuse=self.reuse, is_training=self.is_training,
                                  norm=self.norm, activation=tf.nn.relu, name='conv3_1')
            conv3 = layer.c3s1ksame(conv3, reuse=self.reuse, is_training=self.is_training,
                                    norm=self.norm, activation=tf.nn.relu, name='conv3_2')
            pool3 = tf.nn.max_pool(conv3, [1, 2, 2, 1], [1, 2, 2, 1],
                                   padding='SAME', name='pool3')
            conv4 = layer.c3s1kx2(pool3, reuse=self.reuse, is_training=self.is_training,
                                  norm=self.norm, activation=tf.nn.relu, name='conv4_1')
            conv4 = layer.c3s1ksame(conv4, reuse=self.reuse, is_training=self.is_training,
                                    norm=self.norm, activation=tf.nn.relu, name='conv4_2')
            pool4 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1],
                                   padding='SAME', name='pool4')
            conv5 = layer.c3s1kx2(pool4, reuse=self.reuse, is_training=self.is_training,
                                  norm=self.norm, activation=tf.nn.relu, name='conv5_1')
            conv5 = layer.c3s1ksame(conv5, reuse=self.reuse, is_training=self.is_training,
                                    norm=self.norm, activation=tf.nn.relu, name='conv5_2')

            up6 = layer.upc2s1kd2(conv5, reuse=self.reuse, is_training=self.is_training,
                                  norm=self.norm, activation=tf.nn.relu, name='up6')
            concat6 = tf.concat(values=[conv4, up6], axis=3)
            conv6 = layer.c3s1kd2(concat6, reuse=self.reuse, is_training=self.is_training,
                                  norm=self.norm, activation=tf.nn.relu, name='conv6_1')
            conv6 = layer.c3s1ksame(conv6, reuse=self.reuse, is_training=self.is_training,
                                    norm=self.norm, activation=tf.nn.relu, name='conv6_2')
            up7 = layer.upc2s1kd2(conv6, reuse=self.reuse, is_training=self.is_training,
                                  norm=self.norm, activation=tf.nn.relu, name='up7')
            concat7 = tf.concat(values=[conv3, up7], axis=3)
            conv7 = layer.c3s1kd2(concat7, reuse=self.reuse, is_training=self.is_training,
                                  norm=self.norm, activation=tf.nn.relu, name='conv7_1')
            conv7 = layer.c3s1ksame(conv7, reuse=self.reuse, is_training=self.is_training,
                                    norm=self.norm, activation=tf.nn.relu, name='conv7_2')
            up8 = layer.upc2s1kd2(conv7, reuse=self.reuse, is_training=self.is_training,
                                  norm=self.norm, activation=tf.nn.relu, name='up8')
            concat8 = tf.concat(values=[conv2, up8], axis=3)
            conv8 = layer.c3s1kd2(concat8, reuse=self.reuse, is_training=self.is_training,
                                  norm=self.norm, activation=tf.nn.relu, name='conv8_1')
            conv8 = layer.c3s1ksame(conv8, reuse=self.reuse, is_training=self.is_training,
                                    norm=self.norm, activation=tf.nn.relu, name='conv8_2')
            up9 = layer.upc2s1kd2(conv8, reuse=self.reuse, is_training=self.is_training,
                                  norm=self.norm, activation=tf.nn.relu, name='up9')
            concat9 = tf.concat(values=[conv1, up9], axis=3)
            conv9 = layer.c3s1kd2(concat9, reuse=self.reuse, is_training=self.is_training,
                                  norm=self.norm, activation=tf.nn.relu, name='conv9_1')
            conv9 = layer.c3s1ksame(conv9, reuse=self.reuse, is_training=self.is_training,
                                    norm=self.norm, activation=tf.nn.relu, name='conv9_2')
            conv9 = layer.c3s1k2(conv9, reuse=self.reuse, is_training=self.is_training,
                                 norm=self.norm, activation=tf.nn.relu, name='conv9_3')
            conv10 = layer.c1s1k1(conv9, reuse=self.reuse, is_training=self.is_training,
                                  norm=self.norm, activation=None, name='conv10')
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
            self.output = conv10

            self.loss = tf.reduce_mean(tf.square(self.output - self.y))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate). \
                minimize(self.loss, var_list=self.variables)



    def model(self):
        tf.summary.scalar("loss", self.loss)

        tf.summary.image('original image', self.x)
        tf.summary.image('predicted boundary', utils.batch_convert2int(self.output))
        tf.summary.image('ground truth', self.y)


    """
    def loss_function(self):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.y)
        return loss

    def optimize(self):
        loss = self.loss_function()
        opt = tf.train.AdamOptimizer(self.learning_rate). \
            minimize(loss, var_list=self.variables)
        self.optimizer = opt
        return opt
        """

