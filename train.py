import tensorflow as tf
from model import Unet
from reader import Reader
from datetime import datetime
import logging
import os

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('data', 'data/tfrecords/train.tfrecords',
                       'data input dir, default: data/tfrecords/train.tfrecords')
tf.flags.DEFINE_integer('batch_size', 2, 'batch_size, default : 2')
tf.flags.DEFINE_integer('image_size', 256, 'image size, default: 256')
tf.flags.DEFINE_string('norm', 'batch',
                       '[instance, batch] use instance norm or batch norm, default: batch')
tf.flags.DEFINE_float('learning_rate', 1e-2,
                      'initial learning rate for Adam, default: 0.01')
tf.flags.DEFINE_string('load_model', None,
                        'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')


def train():
    if FLAGS.load_model is not None:
        checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/" + current_time.lstrip("checkpoints/")
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass

    graph = tf.Graph()
    with graph.as_default():
        u_net = Unet('Unet',
                     image_size=FLAGS.image_size,
                     norm=FLAGS.norm,
                     learning_rate=FLAGS.learning_rate)

        loss = u_net.loss
        optimizer = u_net.optimizer
        u_net.model()

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()
        reader = Reader(FLAGS.data, batch_size=FLAGS.batch_size)
        image_train, image_label = reader.feed()

    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        if FLAGS.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            step = int(meta_graph_path.split("-")[2].split(".")[0])
        else:
            sess.run(tf.global_variables_initializer())
            step = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop() and step < 100:
                _train, _label = sess.run([image_train, image_label])
                ls, op, summary= sess.run([loss, optimizer, summary_op], feed_dict={u_net.x : _train,
                                                                u_net.y : _label})
                train_writer.add_summary(summary, step)
                train_writer.flush()

                if (step + 1) % 5 == 0:
                    logging.info('-----------Step %d:-------------' % step)
                    logging.info('    loss    :{}'.format(ls))

                if (step + 1) % 25 == 0:
                    save_path = saver.save(sess, checkpoints_dir + "/model.ckpt",
                                           global_step=step)
                    logging.info("Model saved in file: %s" % save_path)

                step += 1

        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            logging.info("Model saved in file: %s" % save_path)
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

def main(unused_argv):
    train()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()


