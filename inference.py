# !/usr/bin/env python
# coding: utf-8
import os
import logging
from utils.iterator import InferenceIterator
from utils.funcs import prepare_batch, load_inverse_dict, seq2words
import json
import tensorflow as tf
from models.seq2seq import Seq2SeqModel

# Decoding parameters
tf.app.flags.DEFINE_integer('beam_width', 1, 'Beam width used in beam search')
tf.app.flags.DEFINE_integer('inference_batch_size', 4, 'Batch size used for decoding')
tf.app.flags.DEFINE_integer('max_inference_step', 60, 'Maximum time step limit to decode')
tf.app.flags.DEFINE_string('model_path', 'checkpoints/couplet_seq2seq/couplet.ckpt-70000',
                           'Path to a specific model checkpoint.')
tf.app.flags.DEFINE_string('inference_input', 'dataset/couplet/test.x.txt', 'Decoding input path')
tf.app.flags.DEFINE_string('inference_output', 'dataset/couplet/test.inference.txt', 'Decoding output path')

# Runtime parameters
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')
tf.app.flags.DEFINE_string('gpu', '0', 'GPU Number')
tf.app.flags.DEFINE_boolean('debug', True, 'Enable debug mode')
tf.app.flags.DEFINE_string('logger_name', 'train', 'Logger name')
tf.app.flags.DEFINE_string('logger_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 'Logger format')

FLAGS = tf.app.flags.FLAGS

logging_level = logging.DEBUG if FLAGS.debug else logging.INFO
logging.basicConfig(level=logging_level, format=FLAGS.logger_format)
logger = logging.getLogger(FLAGS.logger_name)


def load_config(FLAGS):
    config = json.load(open('%s.json' % FLAGS.model_path, 'r'))
    for key, value in FLAGS.flag_values_dict().items():
        config[key] = value
    return config


def load_model(session, config):
    model = Seq2SeqModel(config, 'inference', logger)
    if tf.train.checkpoint_exists(FLAGS.model_path):
        print('Reloading model parameters..')
        model.restore(session, FLAGS.model_path)
    else:
        raise ValueError(
            'No such file:[{}]'.format(FLAGS.model_path))
    return model


def decode():
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    
    # Load model config
    config = load_config(FLAGS)
    print(config)
    # Load source data to decode
    test_set = InferenceIterator(source=config['inference_input'],
                                 split_sign=config['split_sign'],
                                 batch_size=config['inference_batch_size'],
                                 source_dict=config['source_vocabulary'],
                                 n_words_source=config['encoder_vocab_size'])
    
    # Load inverse dictionary used in decoding
    target_inverse_dict = load_inverse_dict(config['target_vocabulary'])
    
    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                          log_device_placement=FLAGS.log_device_placement,
                                          gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        # Reload existing checkpoint
        model = load_model(sess, config)
        print('Decoding {}..'.format(FLAGS.inference_input))
        
        fout = [open(FLAGS.inference_output, 'w')]
        
        line_number = 0
        
        for idx, source_seq in enumerate(test_set.next()):
            source, source_len = prepare_batch(source_seq)
            line_number += len(source)
            
            print('Source', source[0], 'Source Len', source_len[0])
            
            probabilities, predicts = model.inference(sess, source, source_len)
            
            o = predicts
            
            print('O', o)
            print('O', o.shape)
            
            # print(probs.shape)
            for t, s in zip(o, source):
                print('t', t)
                t = seq2words(t, inverse_target_dictionary=target_inverse_dict)
                s = seq2words(s, inverse_target_dictionary=target_inverse_dict)
                print('s', s, 't', t)
            # print(predicts.shape)
            
            # print(predicts[:5])
            print(o[:5])


def main(_):
    decode()


if __name__ == '__main__':
    tf.app.run()
