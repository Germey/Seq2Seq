import os

model_dir = '/home/mml/siamese_net/logs/train/'
from tensorflow.python import pywrap_tensorflow

# checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
checkpoint_path = os.path.join(model_dir, "model.ckpt-9999")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))
