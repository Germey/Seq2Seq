# !/usr/bin/env python
# coding: utf-8
import os
import math
import time
import json
import numpy as np
import tensorflow as tf
from os.path import join
from utils.iterator import TrainTextIterator, ExtendTextIterator
from models import *
from tqdm import tqdm
from utils.funcs import prepare_pair_batch, get_summary
import os
import logging

# Data loading parameters

tf.app.flags.DEFINE_string('source_vocabulary', 'dataset/couplet/vocab.json', 'Path to source vocabulary')
tf.app.flags.DEFINE_string('target_vocabulary', 'dataset/couplet/vocab.json', 'Path to target vocabulary')
tf.app.flags.DEFINE_string('source_train_data', 'dataset/couplet/train.x.txt',
                           'Path to source training data')
tf.app.flags.DEFINE_string('target_train_data', 'dataset/couplet/train.y.txt',
                           'Path to target training data')
tf.app.flags.DEFINE_string('source_valid_data', 'dataset/couplet/valid.x.txt',
                           'Path to source validation data')
tf.app.flags.DEFINE_string('target_valid_data', 'dataset/couplet/valid.y.txt',
                           'Path to target validation data')

# Network parameters
tf.app.flags.DEFINE_string('model_class', 'seq2seq_attention', 'Model class')
tf.app.flags.DEFINE_string('cell_type', 'gru', 'RNN cell for encoder and decoder, default: lstm')
tf.app.flags.DEFINE_string('attention_type', 'bahdanau', 'Attention mechanism: (bahdanau, luong), default: bahdanau')
tf.app.flags.DEFINE_integer('hidden_units', 500, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('attention_units', 256, 'Number of attention units in each layer')
tf.app.flags.DEFINE_integer('encoder_depth', 3, 'Number of layers in encoder')
tf.app.flags.DEFINE_integer('decoder_depth', 3, 'Number of layers in decoder')
tf.app.flags.DEFINE_integer('embedding_size', 300, 'Embedding dimensions of encoder and decoder inputs')
tf.app.flags.DEFINE_integer('encoder_vocab_size', 6622, 'Source vocabulary size')
tf.app.flags.DEFINE_integer('decoder_vocab_size', 6622, 'Target vocabulary size')
tf.app.flags.DEFINE_boolean('use_residual', False, 'Use residual connection between layers')
tf.app.flags.DEFINE_boolean('use_dropout', True, 'Use dropout in each rnn cell')
tf.app.flags.DEFINE_boolean('use_bidirectional', False, 'Use bidirectional rnn cell')
tf.app.flags.DEFINE_float('dropout_rate', 0.3, 'Dropout probability for input/output/state units (0.0: no dropout)')
tf.app.flags.DEFINE_string('split_sign', ' ', 'Separator of dataset')

# Training parameters
tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')
tf.app.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
tf.app.flags.DEFINE_integer('batch_size', 5, 'Batch size')
tf.app.flags.DEFINE_integer('max_epochs', 10000, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('max_load_batches', 20, 'Maximum # of batches to load at one time')
tf.app.flags.DEFINE_integer('encoder_max_time_steps', 30, 'Maximum sequence length')
tf.app.flags.DEFINE_integer('decoder_max_time_steps', 30, 'Maximum sequence length')
tf.app.flags.DEFINE_integer('display_freq', 5, 'Display training status every this iteration')
tf.app.flags.DEFINE_integer('save_freq', 1000, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_integer('valid_freq', 20, 'Evaluate model every this iteration: valid_data needed')
tf.app.flags.DEFINE_string('optimizer_type', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop)')
tf.app.flags.DEFINE_string('model_dir', 'checkpoints/couplet', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'model.ckpt', 'File name used for model checkpoints')
tf.app.flags.DEFINE_boolean('use_fp16', False, 'Use half precision float16 instead of float32 as dtype')
tf.app.flags.DEFINE_boolean('shuffle_each_epoch', False, 'Shuffle training dataset for each epoch')
tf.app.flags.DEFINE_boolean('sort_by_length', False, 'Sort pre-fetched mini batches by their target sequence lengths')

# Runtime parameters
tf.app.flags.DEFINE_string('gpu', '-1', 'GPU number')
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')
tf.app.flags.DEFINE_boolean('debug', True, 'Enable debug mode')
tf.app.flags.DEFINE_string('logger_name', 'train', 'Logger name')
tf.app.flags.DEFINE_string('logger_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s', 'Logger format')

FLAGS = tf.app.flags.FLAGS

logging_level = logging.DEBUG if FLAGS.debug else logging.INFO
logging.basicConfig(level=logging_level, format=FLAGS.logger_format)
logger = logging.getLogger(FLAGS.logger_name)


def get_model_class():
    model_class = FLAGS.model_class
    class_map = {
        'seq2seq': Seq2SeqModel,
        'seq2seq_attention': Seq2SeqAttentionModel,
        'pointer_generator': PointerGeneratorModel
    }
    assert model_class in class_map.keys()
    return class_map[model_class]


def create_model(session, FLAGS):
    config = FLAGS.flag_values_dict()
    model_class = get_model_class()
    model = model_class(config, 'train', logger)
    
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info('Reloading model parameters..')
        model.restore(session, ckpt.model_checkpoint_path)
    
    else:
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        logger.info('Created new model parameters..')
        session.run(tf.global_variables_initializer())
    
    return model


def train():
    if int(FLAGS.gpu) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    logger.info('Using GPU %s', os.environ.get('CUDA_VISIBLE_DEVICES'))
    # Load parallel data to train
    logger.info('Loading training data...')
    train_set = TrainTextIterator(source=FLAGS.source_train_data,
                                  target=FLAGS.target_train_data,
                                  source_dict=FLAGS.source_vocabulary,
                                  target_dict=FLAGS.target_vocabulary,
                                  batch_size=FLAGS.batch_size,
                                  n_words_source=FLAGS.encoder_vocab_size,
                                  n_words_target=FLAGS.decoder_vocab_size,
                                  sort_by_length=FLAGS.sort_by_length,
                                  split_sign=FLAGS.split_sign,
                                  max_length=None,
                                  )
    if FLAGS.model_class == 'pointer_generator':
        train_set = ExtendTextIterator(source=FLAGS.source_train_data,
                                       target=FLAGS.target_train_data,
                                       source_dict=FLAGS.source_vocabulary,
                                       target_dict=FLAGS.target_vocabulary,
                                       batch_size=FLAGS.batch_size,
                                       n_words_source=FLAGS.encoder_vocab_size,
                                       n_words_target=FLAGS.decoder_vocab_size,
                                       sort_by_length=FLAGS.sort_by_length,
                                       split_sign=FLAGS.split_sign,
                                       max_length=None,
                                       )
    
    if FLAGS.source_valid_data and FLAGS.target_valid_data:
        logger.info('Loading validation data...')
        valid_set = TrainTextIterator(source=FLAGS.source_valid_data,
                                      target=FLAGS.target_valid_data,
                                      source_dict=FLAGS.source_vocabulary,
                                      target_dict=FLAGS.target_vocabulary,
                                      batch_size=FLAGS.batch_size,
                                      n_words_source=FLAGS.encoder_vocab_size,
                                      n_words_target=FLAGS.decoder_vocab_size,
                                      sort_by_length=FLAGS.sort_by_length,
                                      split_sign=FLAGS.split_sign,
                                      max_length=None
                                      )
        if FLAGS.model_class == 'pointer_generator':
            valid_set = ExtendTextIterator(source=FLAGS.source_valid_data,
                                           target=FLAGS.target_valid_data,
                                           source_dict=FLAGS.source_vocabulary,
                                           target_dict=FLAGS.target_vocabulary,
                                           batch_size=FLAGS.batch_size,
                                           n_words_source=FLAGS.encoder_vocab_size,
                                           n_words_target=FLAGS.decoder_vocab_size,
                                           sort_by_length=FLAGS.sort_by_length,
                                           split_sign=FLAGS.split_sign,
                                           max_length=None
                                           )
    else:
        valid_set = None
    
    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                          log_device_placement=FLAGS.log_device_placement,
                                          gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        
        # Create a new model or reload existing checkpoint
        model = create_model(sess, FLAGS)
        
        # Create a log writer object
        train_summary_writer = tf.summary.FileWriter(join(FLAGS.model_dir, 'train'), graph=sess.graph)
        valid_summary_writer = tf.summary.FileWriter(join(FLAGS.model_dir, 'valid'), graph=sess.graph)
        
        step_time, loss = 0.0, 0.0
        words_seen, sents_seen, processed_number = 0, 0, 0
        start_time = time.time()
        
        # Training loop
        logger.info('Training...')
        
        for epoch_idx in range(FLAGS.max_epochs):
            if model.global_epoch_step.eval() >= FLAGS.max_epochs:
                logger.info('Training is already complete. current epoch: %s, max epoch: %s',
                            model.global_epoch_step.eval(), FLAGS.max_epochs)
                break
            
            train_set.reset()
            
            with tqdm(total=train_set.length()) as pbar:
                
                for batch in train_set.next():
                    
                    if FLAGS.model_class == 'pointer_generator':
                        source_batch, target_batch, source_extend_batch, target_extend_batch, oovs_max_size = batch
                        
                        # Get a batch from training parallel data
                        source, source_len, target, target_len = prepare_pair_batch(
                            source_batch, target_batch,
                            FLAGS.encoder_max_time_steps,
                            FLAGS.decoder_max_time_steps)
                        
                        # Get a batch from training parallel data
                        source_extend, _, target_extend, _ = prepare_pair_batch(
                            source_extend_batch, target_extend_batch,
                            FLAGS.encoder_max_time_steps,
                            FLAGS.decoder_max_time_steps)
                        logger.info('Training batch data shape %s, %s, %s, %s', source.shape, target.shape,
                                    source_extend.shape, target_extend.shape)
                        processed_number += len(source_batch)
                        
                        # Execute a single training step
                        step_loss, _ = model.train(sess, encoder_inputs=source,
                                                   encoder_inputs_extend=source_extend,
                                                   encoder_inputs_length=source_len,
                                                   decoder_inputs=target,
                                                   decoder_inputs_extend=target_extend,
                                                   decoder_inputs_length=target_len,
                                                   oovs_max_size=oovs_max_size
                                                   )
                    
                    else:
                        source_batch, target_batch = batch
                        
                        # Get a batch from training parallel data
                        source, source_len, target, target_len = prepare_pair_batch(source_batch, target_batch,
                                                                                    FLAGS.encoder_max_time_steps,
                                                                                    FLAGS.decoder_max_time_steps)
                        logger.info('Training batch data shape %s, %s', source.shape, target.shape)
                        
                        processed_number += len(source_batch)
                        
                        # Execute a single training step
                        step_loss, _ = model.train(sess, encoder_inputs=source, encoder_inputs_length=source_len,
                                                   decoder_inputs=target, decoder_inputs_length=target_len)
                    
                    loss += float(step_loss) / FLAGS.display_freq
                    
                    words_seen += float(np.sum(source_len + target_len))
                    sents_seen += float(source.shape[0])  # batch_size
                    
                    if model.global_step.eval() % FLAGS.display_freq == 0:
                        avg_perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                        
                        time_elapsed = time.time() - start_time
                        step_time = time_elapsed / FLAGS.display_freq
                        
                        words_per_sec = words_seen / time_elapsed
                        sents_per_sec = sents_seen / time_elapsed
                        
                        logger.info(
                            'Epoch: %s Step: %s Perplexity: %.2f Loss: %s Step-time: %s %.2f sents/s %.2f words/s',
                            model.global_epoch_step.eval(),
                            model.global_step.eval(),
                            avg_perplexity,
                            loss,
                            step_time,
                            sents_per_sec,
                            words_per_sec
                        )
                        
                        # Record training summary for the current batch
                        summary = get_summary('train_loss', loss)
                        train_summary_writer.add_summary(summary, model.global_step.eval())
                        logger.info('Recording training summary step: %s', model.global_step.eval())
                        train_summary_writer.flush()
                        
                        # logger.info('Processed Number', processed_number)
                        pbar.update(processed_number)
                        
                        loss = 0
                        words_seen = 0
                        sents_seen = 0
                        processed_number = 0
                        start_time = time.time()
                    
                    # Execute a validation step
                    if valid_set and model.global_step.eval() % FLAGS.valid_freq == 0:
                        logger.info('Validating...')
                        valid_loss = 0.0
                        valid_sents_seen = 0
                        
                        valid_set.reset()
                        
                        for batch in valid_set.next():
                            
                            if FLAGS.model_class == 'pointer_generator':
                                source_batch, target_batch, source_extend_batch, target_extend_batch, oovs_max_size = batch
                                
                                # Get a batch from training parallel data
                                source, source_len, target, target_len = prepare_pair_batch(
                                    source_batch, target_batch,
                                    FLAGS.encoder_max_time_steps,
                                    FLAGS.decoder_max_time_steps)
                                
                                # Get a batch from training parallel data
                                source_extend, _, target_extend, _ = prepare_pair_batch(
                                    source_extend_batch, target_extend_batch,
                                    FLAGS.encoder_max_time_steps,
                                    FLAGS.decoder_max_time_steps)
                                logger.info('Training batch data shape %s, %s, %s, %s', source.shape, target.shape,
                                            source_extend.shape, target_extend.shape)
                                processed_number += len(source_batch)
                                
                                # Execute a single training step
                                step_loss = model.eval(sess, encoder_inputs=source,
                                                       encoder_inputs_extend=source_extend,
                                                       encoder_inputs_length=source_len,
                                                       decoder_inputs=target,
                                                       decoder_inputs_extend=target_extend,
                                                       decoder_inputs_length=target_len,
                                                       oovs_max_size=oovs_max_size
                                                       )
                            
                            else:
                                source_batch, target_batch = batch
                                
                                # Get a batch from training parallel data
                                source, source_len, target, target_len = prepare_pair_batch(source_batch, target_batch,
                                                                                            FLAGS.encoder_max_time_steps,
                                                                                            FLAGS.decoder_max_time_steps)
                                logger.info('Training batch data shape %s, %s', source.shape, target.shape)
                                
                                processed_number += len(source_batch)
                                
                                # Execute a single training step
                                step_loss = model.eval(sess, encoder_inputs=source,
                                                       encoder_inputs_length=source_len,
                                                       decoder_inputs=target, decoder_inputs_length=target_len)
                            
                            batch_size = source.shape[0]
                            
                            valid_loss += step_loss * batch_size
                            valid_sents_seen += batch_size
                            logger.info('%s samples seen', valid_sents_seen)
                        
                        valid_loss = valid_loss / valid_sents_seen
                        logger.info('Valid perplexity: %.2f Loss: %s', math.exp(valid_loss), valid_loss)
                        
                        # Record training summary for the current batch
                        summary = get_summary('valid_loss', valid_loss)
                        valid_summary_writer.add_summary(summary, model.global_step.eval())
                        logger.info('Recording valid summary step: %s', model.global_step.eval())
                        valid_summary_writer.flush()
                    
                    # Save the model checkpoint
                    if model.global_step.eval() % FLAGS.save_freq == 0:
                        logger.info('Saving the model...')
                        checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                        model.save(sess, checkpoint_path, global_step=model.global_step)
                        json.dump(model.config,
                                  open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'w'),
                                  indent=2)
            
            # Increase the epoch index of the model
            model.global_epoch_step_op.eval()
            logger.info('Epoch %s DONE', model.global_epoch_step.eval())
        
        logger.info('Saving the last model...')
        checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
        model.save(sess, checkpoint_path, global_step=model.global_step)
        json.dump(model.config,
                  open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'w', encoding='utf-8'),
                  indent=2)
    
    logger.info('Training Terminated')


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
