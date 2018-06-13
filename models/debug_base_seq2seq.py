import tensorflow as tf

from utils.config import GO, EOS


class Seq2SeqModel():
    def __init__(self, config, mode, logger):
        assert mode.lower() in ['train', 'inference']
        self.mode = mode.lower()
        self.logger = logger
        self.init_config(config)
        self.build_placeholders()
        self.build_embed()
        self.build_encoder()
        self.build_decoder()
        self.build_optimizer()
    
    def init_config(self, config):
        self.config = config
        self.hidden_units = config['hidden_units']
        self.embedding_size = config['embedding_size']
        self.encoder_max_time_steps = config['decoder_max_time_steps']
        self.decoder_max_time_steps = config['decoder_max_time_steps']
        self.encoder_depth = config['encoder_depth']
        self.decoder_depth = config['decoder_depth']
        self.encoder_vocab_size = config['encoder_vocab_size']
        self.decoder_vocab_size = config['decoder_vocab_size']
        self.logger_name = config['logger_name']
        self.dropout_rate = config['dropout_rate']
        self.dtype = tf.float16 if config['use_fp16'] else tf.float32
        self.optimizer_type = config['optimizer_type']
        self.learning_rate = config['learning_rate']
        self.max_gradient_norm = config['max_gradient_norm']
        self.use_bidirectional = config['use_bidirectional']
        self.use_dropout = config['use_dropout']
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, tf.add(self.global_epoch_step, 1))
    
    def build_placeholders(self):
    
        self.keep_prob = tf.placeholder(self.dtype, shape=[], name='keep_prob')
        
        # encoder_inputs: [batch_size, max_time_steps]
        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                             name='encoder_inputs')
        self.logger.debug('encoder_inputs %s', self.encoder_inputs)
        
        # encoder_inputs_length: [batch_size]
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=[None],
                                                    name='encoder_inputs_length')
        self.logger.debug('encoder_inputs_length %s', self.encoder_inputs_length)
        
        # batch_size
        self.batch_size = tf.shape(self.encoder_inputs)[0]
        self.logger.debug('batch_size %s', self.batch_size)
        
        if self.mode == 'train':
            
            # decoder_inputs: [batch_size, max_time_steps]
            self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                                 name='decoder_inputs')
            self.logger.debug('decoder_inputs %s', self.decoder_inputs)
            
            # decoder_inputs_length: [batch_size]
            self.decoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=[None],
                                                        name='decoder_inputs_length')
            self.logger.debug('decoder_inputs_length %s', self.decoder_inputs_length)
            
            # decoder_start_token: [batch_size, 1]
            self.decoder_start_token = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * GO
            self.logger.debug('decoder_start_token %s', self.decoder_start_token)
            
            # decoder_end_token: [batch_size, 1]
            self.decoder_end_token = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * EOS
            self.logger.debug('decoder_end_token %s', self.decoder_end_token)
            
            # decoder_inputs_train: [batch_size, max_time_steps + 1]
            self.decoder_inputs_train = tf.concat([self.decoder_start_token, self.decoder_inputs], axis=-1)
            self.logger.debug('decoder_inputs_train %s', self.decoder_inputs_train)
            
            # decoder_inputs_train_length: [batch_size]
            self.decoder_inputs_train_length = self.decoder_inputs_length + 1
            self.logger.debug('decoder_inputs_train_length %s', self.decoder_inputs_train_length)
            
            # decoder_targets_train: [batch_size, max_time_steps + 1]
            self.decoder_targets_train = tf.concat([self.decoder_inputs, self.decoder_end_token], axis=-1)
            self.logger.debug('decoder_targets_train %s', self.decoder_targets_train)
            
            # decoder_targets_length: [batch_size]
            self.decoder_targets_train_length = self.decoder_inputs_length + 1
            self.logger.debug('decoder_targets_train_length %s', self.decoder_targets_train_length)
        
        else:
            self.decoder_inputs = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32, name='decoder_inputs') * GO
            self.logger.debug('decoder_inputs %s', self.decoder_inputs)
            
            self.decoder_inputs_inference = self.decoder_inputs
            self.logger.debug('decoder_inputs_inference %s', self.decoder_inputs_inference)
            
            self.decoder_inputs_inference_length = tf.ones(shape=[self.batch_size], dtype=tf.int32,
                                                           name='decoder_inputs_inference_length')
            self.logger.debug('decoder_inputs_inference_length %s', self.decoder_inputs_inference_length)

        self.max_sequence_length = tf.reduce_max(self.decoder_inputs_train_length, name='max_length')
        self.mask = tf.sequence_mask(self.decoder_inputs_train_length, self.max_sequence_length, dtype=tf.float32,
                                     name='masks')

    def build_single_cell(self):
        """
        build single cell, lstm or gru or RNN
        :return: GRUCell or LSTMCell or RNNCell
        """
        cell = tf.nn.rnn_cell.GRUCell(self.hidden_units, name='single_cell')
        if self.use_dropout:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, dtype=self.dtype, output_keep_prob=self.keep_prob)
        return cell
    
    def build_encoder_cell(self, depth=None):
        """
        build encoder multi cell
        :return: MultiRNNCell
        """
        depth = depth if depth else self.encoder_depth
        cells = [self.build_single_cell() for _ in range(depth)]
        return tf.nn.rnn_cell.MultiRNNCell(cells=cells)

    def create_bi_rnn(self, inputs, scope='bi_rnn', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_units)
            cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_units)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, dtype=tf.float32)
        return outputs, states

    def build_embed(self):
        self.lookup_table = tf.get_variable('lookup_table', dtype=tf.float32,
                                            shape=[self.encoder_vocab_size, self.embedding_size],
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))

    def build_encoder(self):
        encoder_enc = tf.nn.embedding_lookup(self.lookup_table, self.encoder_inputs)
        outputs_1, states_1 = self.create_bi_rnn(encoder_enc, scope='encoder_birnn_1')
        outputs_1 = tf.concat(outputs_1, -1)
        outputs_2, states_2 = self.create_bi_rnn(outputs_1, scope='encoder_birnn_2')
        forward_state, backward_state = states_2
        self.encoder_last_state = backward_state
    
    def build_decoder_cell(self, depth=None):
        depth = depth if depth else self.decoder_depth
        cells = [self.build_single_cell() for _ in range(depth)]
        return tf.nn.rnn_cell.MultiRNNCell(cells=cells)
    
    def build_decoder(self):
        decoder_enc = tf.nn.embedding_lookup(self.lookup_table, self.decoder_inputs_train)
        decoder_cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
    
        self.initial_state = self.encoder_last_state
        logits, states = tf.nn.dynamic_rnn(decoder_cell, decoder_enc,
                                           initial_state=self.initial_state,
                                           sequence_length=self.decoder_inputs_train_length, dtype=tf.float32)
        self.last_state = states
        self.outputs = tf.layers.dense(logits, self.encoder_vocab_size, use_bias=True)
        self.probs = tf.nn.softmax(self.outputs, -1)
        self.preds = tf.argmax(self.probs, -1)
    
        # training settings
        if self.mode == 'train':
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.outputs,
                                                         targets=self.decoder_targets_train, weights=self.mask)
            
        
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
    
    def build_optimizer(self):
        if self.mode == 'train':
            self.logger.info('Setting optimizer...')
            
            # trainable_verbs
            self.trainable_verbs = tf.trainable_variables()
            # self.logger.debug('trainable_verbs %s', self.trainable_verbs)
            
            if self.optimizer_type.lower() == 'adam':
                self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
                self.logger.info('Optimizer has been set')
            
            # compute gradients
            # self.gradients = tf.gradients(ys=self.loss, xs=self.trainable_verbs)
            
            # clip gradients by a given maximum_gradient_norm
            # self.clip_gradients, _ = tf.clip_by_global_norm(self.gradients, self.max_gradient_norm)
            
            # train op
            # self.train_op = self.optimizer.apply_gradients(zip(self.clip_gradients, self.trainable_verbs),
            #                                               global_step=self.global_step)
            
            self.train_op = self.optimizer.minimize(loss=self.loss, global_step=self.global_step)
    
    def save(self, sess, save_path, var_list=None, global_step=None):
        saver = tf.train.Saver(var_list)
        
        # save model
        saver.save(sess=sess, save_path=save_path, global_step=global_step)
        self.logger.info('model saved at %s', save_path)
    
    def restore(self, sess, save_path, var_list=None):
        saver = tf.train.Saver(var_list)
        saver.restore(sess=sess, save_path=save_path)
        self.logger.info('model restored from %s', save_path)
    
    def train(self, sess, encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length):
        
        input_feed = {
            self.encoder_inputs.name: encoder_inputs,
            self.encoder_inputs_length.name: encoder_inputs_length,
            self.decoder_inputs.name: decoder_inputs,
            self.decoder_inputs_length.name: decoder_inputs_length,
            self.keep_prob.name: 1 - self.dropout_rate
        }
        
        output_feed = [
            self.loss,
            self.train_op,
        ]
        outputs = sess.run(fetches=output_feed, feed_dict=input_feed)
        return outputs
    
    def eval(self, sess, encoder_inputs, encoder_inputs_length,
             decoder_inputs, decoder_inputs_length):
        
        input_feed = {
            self.encoder_inputs.name: encoder_inputs,
            self.encoder_inputs_length.name: encoder_inputs_length,
            self.decoder_inputs.name: decoder_inputs,
            self.decoder_inputs_length.name: decoder_inputs_length,
            self.keep_prob.name: 1
        }
        
        output_feed = self.loss
        
        outputs = sess.run(fetches=output_feed, feed_dict=input_feed)
        return outputs
    
    def inference(self, sess, encoder_inputs, encoder_inputs_length):
        input_feed = {
            self.encoder_inputs.name: encoder_inputs,
            self.encoder_inputs_length.name: encoder_inputs_length,
            self.keep_prob.name: 1
        }
        
        output_feed = [
            self.decoder_probabilities,
            self.decoder_predicts,
        ]
        outputs = sess.run(fetches=output_feed, feed_dict=input_feed)
        return outputs
