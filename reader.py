import tensorflow as tf
import utils
import matplotlib.pyplot as plt
import numpy as np

class Reader:
    def __init__(self, tfrecords_file, image_size=256, min_queue_examples=30, batch_size=2,
                 num_threads=12, name=''):
        self.tfrecords_file = tfrecords_file
        self.image_size = image_size
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.name = name
        self.reader = tf.TFRecordReader()

    def feed(self):
        with tf.name_scope(self.name):
            filename_queue = tf.train.string_input_producer([self.tfrecords_file])
            reader = tf.TFRecordReader()

            _, serialized_example = self.reader.read(filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image/train': tf.FixedLenFeature([], tf.string),
                    'image/label': tf.FixedLenFeature([], tf.string),
                })

            train_buffer = features['image/train']
            label_buffer = features['image/label']
            train = tf.image.decode_jpeg(train_buffer, channels=1)
            label = tf.image.decode_jpeg(label_buffer, channels=1)
            train = self._preprocess(train)
            label = self._preprocess(label)
            train, label = tf.train.shuffle_batch(
                [train, label], batch_size=self.batch_size, num_threads=self.num_threads,
                capacity=self.min_queue_examples + 3 * self.batch_size,
                min_after_dequeue=self.min_queue_examples
            )

            #tf.summary.image('_input', train)
        return train, label

    def _preprocess(self, image):
        image = tf.image.resize_images(image, size=(self.image_size, self.image_size))
        image = utils.convert2float(image)
        image.set_shape([self.image_size, self.image_size, 1])
        return image

def test_reader():
    TRAIN_FILE_1 = 'data/tfrecords/train.tfrecords'

    with tf.Graph().as_default():
        reader1 = Reader(TRAIN_FILE_1, batch_size=1)
        image_train, image_label = reader1.feed()
        #image_train = tf.squeeze(image_train, 0)
        #image_label = tf.squeeze(image_label, 0)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)



        try:
            step = 0
            while not coord.should_stop() and step < 1:
                train, label = sess.run([image_train, image_label])
                print(train.shape, label.shape)
                f, a = plt.subplots(2, 1)
                #for i in range(1):
                a[0].imshow(np.reshape(train, (256, 256)))
                a[1].imshow(np.reshape(label, (256, 256)))
                plt.show()
                step += 1
        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
  test_reader()