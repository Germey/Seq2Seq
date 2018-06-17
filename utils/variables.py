import os
import tensorflow as tf
from tensorflow.python.pywrap_tensorflow_internal import NewCheckpointReader

tf.app.flags.DEFINE_string('model_path', '../checkpoints/couplet_seq2seq', 'Model path')
tf.app.flags.DEFINE_string('model_name', 'couplet.ckpt-70000', 'Model name')
tf.app.flags.DEFINE_bool('print_value', False, 'Print value of tensors')
FLAGS = tf.app.flags.FLAGS

# checkpoint path
checkpoint_path = os.path.join(FLAGS.model_path, FLAGS.model_name)
reader = NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

tensors = []
# collect all keys
for key in var_to_shape_map:
    tensors.append(key)

# print keys and values
for key in sorted(tensors):
    print('tensor_name:', key)
    if FLAGS.print_value:
        print(reader.get_tensor(key))
