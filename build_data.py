import tensorflow as tf
import random
import os
from os import scandir

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('train_input_dir', 'data/train',
                       'train input directory, default: data/train')
tf.flags.DEFINE_string('label_input_dir', 'data/label',
                       'label input directory, default: data/train')
tf.flags.DEFINE_string('output_dir', 'data/tfrecords/train.tfrecords',
                       'output directory, default: data/tfrecords/output')


def data_reader(train_input_dir, label_input_dir, shuffle=True):
    """Read images from input_dir then shuffle them
    Args:
        train_input_dir: string, path of input train dir, e.g., /path/to/dir
        label_input_dir: string, path of input label dir, e.g., /path/to/dir
    Returns:
        file_paths: list of strings
    """
    train_file_paths = []
    label_file_paths = []

    for img_file in scandir(train_input_dir):
        if img_file.name.endswith('.png') and img_file.is_file():
            train_file_paths.append(img_file.path)

    for label_file in scandir(label_input_dir):
        if label_file.name.endswith('.png') and label_file.is_file():
            label_file_paths.append(label_file.path)

    if shuffle:
        # Shuffle the ordering of all image files in order to guarantee
        # random ordering of the images with respect to label in the
        # saved TFRecord files. Make the randomization repeatable.
        shuffled_index = list(range(len(train_file_paths)))
        random.seed(12345)
        random.shuffle(shuffled_index)

        train_file_paths = [train_file_paths[i] for i in shuffled_index]
        label_file_paths = [label_file_paths[i] for i in shuffled_index]

    return train_file_paths, label_file_paths


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(train_data, label_data):
    """Build an Example proto for an example.
    Args:
        train_data: string, JPEG encoding of RGB or GRAY image
        label_data: string, JPEG encoding of RGB or GRAY image
    Returns:
        Example proto
    """

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/train': _bytes_feature(train_data),
        'image/label': _bytes_feature(label_data)
        }))
    return example


def data_writer(train_input_dir, label_input_dir, output_file):
    """Write data to tfrecords
    """
    train_file_paths, label_file_paths = data_reader(train_input_dir, label_input_dir)

    # create tfrecords dir if not exists
    output_dir = os.path.dirname(output_file)
    try:
        os.makedirs(output_dir)
    except os.error:
        pass

    images_num = len(train_file_paths)

    # dump to tfrecords file
    writer = tf.python_io.TFRecordWriter(output_file)

    for i in range(len(train_file_paths)):
        train_file_path = train_file_paths[i]
        label_file_path = label_file_paths[i]

        with tf.gfile.FastGFile(train_file_path, 'rb') as f:
            train_data = f.read()
        with tf.gfile.FastGFile(label_file_path, 'rb') as f:
            label_data = f.read()

        example = _convert_to_example(train_data, label_data)
        writer.write(example.SerializeToString())

        if (i + 1) % 10 == 0:
            print("Processed {}/{}.".format(i + 1, images_num))
    print("Done.")
    writer.close()


def main():
    print("Convert train and label to tfrecords...")
    data_writer(FLAGS.train_input_dir, FLAGS.label_input_dir, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
