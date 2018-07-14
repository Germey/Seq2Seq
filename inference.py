# !/usr/bin/env python
# coding: utf-8
import os
import logging

from preprocess.config import UNK
from utils.iterator import InferenceIterator, ExtendTextIterator, end_token
from utils.funcs import prepare_batch, load_inverse_dict, inverse_dict
import json
import tensorflow as tf
from cls import get_model_class

# Decoding parameters
tf.app.flags.DEFINE_integer('beam_width', 1, 'Beam width used in beam search')
tf.app.flags.DEFINE_integer('inference_batch_size', 256, 'Batch size used for decoding')
tf.app.flags.DEFINE_integer('max_inference_step', 60, 'Maximum time step limit to decode')
tf.app.flags.DEFINE_string('model_path', 'checkpoints/lcsts_word_pointer_generator/lcsts.ckpt-1030000',
                           'Path to a specific model checkpoint.')
tf.app.flags.DEFINE_string('inference_input', 'dataset/lcsts/word/sources.test.txt', 'Decoding input path')
tf.app.flags.DEFINE_string('inference_output', 'dataset/lcsts/char/summaries.inference.txt', 'Decoding output path')

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
    model_class = get_model_class(config['model_class'])
    model = model_class(config, 'inference', logger)
    if tf.train.checkpoint_exists(FLAGS.model_path):
        logger.info('Reloading model parameters..')
        model.restore(session, FLAGS.model_path)
    else:
        raise ValueError(
            'No such file:[{}]'.format(FLAGS.model_path))
    return model


def seq2words(seq, inverse_target_dictionary, oovs_vocab=None):
    words = []
    if oovs_vocab:
        inverse_target_dictionary.update(oovs_vocab)
    for w in seq:
        if w == end_token:
            break
        if w in inverse_target_dictionary:
            result = inverse_target_dictionary[w]
            if result == 'UNK':
                words.append(result)
            else:
                words += list(result)
        else:
            words.append(UNK)
    return ' '.join(words)


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
    
    if config['model_class'].startswith('pointer_generator'):
        test_set = ExtendTextIterator(source=config['inference_input'],
                                      target=config['inference_input'],
                                      source_dict=config['source_vocabulary'],
                                      target_dict=config['target_vocabulary'],
                                      batch_size=config['inference_batch_size'],
                                      n_words_source=config['encoder_vocab_size'],
                                      n_words_target=config['decoder_vocab_size'],
                                      sort_by_length=config['sort_by_length'],
                                      split_sign=config['split_sign'],
                                      max_length=None,
                                      )
    
    test_set.reset()
    
    # Load inverse dictionary used in decoding
    target_inverse_dict = load_inverse_dict(config['target_vocabulary'])
    
    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                          log_device_placement=FLAGS.log_device_placement,
                                          gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        # Reload existing checkpoint
        model = load_model(sess, config)
        logger.info('Decoding %s...', FLAGS.inference_input)
        
        fout = open(FLAGS.inference_output, 'w')
        
        line_number = 0
        
        for idx, batch in enumerate(test_set.next()):
            source_batch, target_batch, source_extend_batch, target_extend_batch, oovs_max_size, oovs_vocabs = batch
            
            source, source_len = prepare_batch(source_batch, config['encoder_max_time_steps'])
            source_extend, _ = prepare_batch(source_extend_batch, config['encoder_max_time_steps'])
            line_number += len(source)
            
            predicts, scores = model.inference(sess,
                                               encoder_inputs=source,
                                               encoder_inputs_extend=source_extend,
                                               encoder_inputs_length=source_len,
                                               oovs_max_size=oovs_max_size)
            
            for predict_seq, score_seq, oovs_vocab in zip(predicts, scores, oovs_vocabs):
                result = seq2words(predict_seq, inverse_target_dictionary=target_inverse_dict,
                                   oovs_vocab=inverse_dict(oovs_vocab))
                logger.info('result %s', result)
                fout.write(result + '\n')
            logger.info('%s lines processed', line_number)
        
        fout.close()
        
        logger.info('Finished!')


def main(_):
    decode()


if __name__ == '__main__':
    tf.app.run()
