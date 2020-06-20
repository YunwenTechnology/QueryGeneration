import tensorflow as tf
import sys
import os
sys.path.append('/iyunwen/lcong/QueryGeneration')

from train.dataloader import input_fn_builder
from train.modeling import model_fn_builder, GroverConfig

flags = tf.flags

FLAGS = flags.FLAGS
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

## Required parameters
flags.DEFINE_string(
    "config_file", '../configs/mega.json',
    "The config json file corresponding to the pre-trained news model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", "",
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", "",
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", "",
    "Initial checkpoint (usually from a pre-trained model).")

flags.DEFINE_integer(
    "max_seq_length", 64,
    "The maximum total input sequence length after BPE tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer("train_batch_size", 6, "Total batch size for training.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for adafactor.")

flags.DEFINE_integer("num_train_steps", 850000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 100000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 20000,
                     "How often to save the model checkpoint.")


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    news_config = GroverConfig.from_json_file(FLAGS.config_file)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Input Files ***")
    for input_file in input_files:
        tf.logging.info("  %s" % input_file)

    my_per_process_gpu_memory_fraction = 1.0
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=my_per_process_gpu_memory_fraction)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    run_config = tf.estimator.RunConfig(model_dir=FLAGS.output_dir,
                                        session_config=sess_config,
                                        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                        keep_checkpoint_max=None)

    model_fn = model_fn_builder(news_config, init_checkpoint=FLAGS.init_checkpoint,
                                learning_rate=FLAGS.learning_rate,
                                num_train_steps=FLAGS.num_train_steps,
                                num_warmup_steps=FLAGS.num_warmup_steps)

    # # If TPU is not available, this will fall back to normal Estimator on CPU
    # # or GPU.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config
    )

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder(
        input_files=input_files,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        batch_size=FLAGS.train_batch_size)

    print("Start trainning.............................................")
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
