import tensorflow as tf
import logging

from tensorflow.python.ops.rnn_cell_impl import MultiRNNCell

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

GO = 0
EOS = 1
UNK = 2


class Seq2SeqModel():
    def __init__(self):
        self.init_config()
        self.init_placeholders()
        self.build_encoder()
        self.build_decoder()
    
    def init_config(self):
        self.batch_size = 256
        self.hidden_units = 128
        self.mode = 'train'
        self.embedding_size = 300
        self.encoder_max_time_steps = 30
        self.decoder_max_time_steps = 30
        self.encoder_depth = 1
        self.decoder_depth = 3
        self.encoder_vocab_size = 8888
        self.decoder_vocab_size = 8888
        self.logger_name = 'main'
        self.dtype = tf.float32
        self.use_bidirectional = True
        self.logger = logging.getLogger(self.logger_name)
    
    def init_placeholders(self):
        # encoder_inputs: [batch_size, max_time_steps]
        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.encoder_max_time_steps],
                                             name='encoder_inputs')
        self.logger.info('encoder_inputs %s', self.encoder_inputs)
        
        # encoder_inputs_length: [batch_size]
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=[self.batch_size],
                                                    name='encoder_inputs_length')
        self.logger.info('encoder_inputs_length %s', self.encoder_inputs_length)
        
        if self.mode == 'train':
            # decoder_inputs: [batch_size, max_time_steps]
            self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.decoder_max_time_steps],
                                                 name='decoder_inputs')
            self.logger.info('decoder_inputs %s', self.decoder_inputs)
            
            # decoder_inputs_length: [batch_size]
            self.decoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=[self.batch_size],
                                                        name='decoder_inputs_length')
            self.logger.info('decoder_inputs_length %s', self.decoder_inputs_length)
            
            # decoder_start_token: [batch_size, 1]
            self.decoder_start_token = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * GO
            self.logger.info('decoder_start_token %s', self.decoder_start_token)
            
            # decoder_end_token: [batch_size, 1]
            self.decoder_end_token = tf.ones(shape=[self.batch_size, 1], dtype=tf.int32) * EOS
            self.logger.info('decoder_end_token %s', self.decoder_end_token)
            
            # decoder_inputs_train: [batch_size, max_time_steps + 1]
            self.decoder_inputs_train = tf.concat([self.decoder_start_token, self.decoder_inputs], axis=-1)
            self.logger.info('decoder_inputs_train %s', self.decoder_inputs_train)
            
            # decoder_inputs_train_length: [batch_size]
            self.decoder_inputs_train_length = self.decoder_inputs_length + 1
            self.logger.info('decoder_inputs_train_length %s', self.decoder_inputs_train_length)
            
            # decoder_targets_train: [batch_size, max_time_steps + 1]
            self.decoder_targets_train = tf.concat([self.decoder_inputs, self.decoder_end_token], axis=-1)
            self.logger.info('decoder_targets_train %s', self.decoder_targets_train)
            
            # decoder_targets_train_length: [batch_size]
            self.decoder_targets_train_length = self.decoder_inputs_length + 1
            self.logger.info('decoder_targets_train_length %s', self.decoder_targets_train_length)
    
    def build_single_cell(self):
        """
        build single cell, lstm or gru or RNN
        :return: GRUCell or LSTMCell or RNNCell
        """
        cell = tf.nn.rnn_cell.GRUCell(self.hidden_units, name='single_cell')
        return cell
    
    def build_encoder_multi_cell(self, depth=None):
        """
        build encoder multi cell
        :return: MultiRNNCell
        """
        depth = depth if depth else self.encoder_depth
        cells = [self.build_single_cell() for _ in range(depth)]
        return tf.nn.rnn_cell.MultiRNNCell(cells=cells)
    
    def build_encoder(self):
        with tf.variable_scope('encoder'):
            # encoder_embeddings: [encoder_vocab_size, embedding_size]
            self.encoder_embeddings = tf.get_variable(name='embedding',
                                                      shape=[self.encoder_vocab_size, self.embedding_size],
                                                      dtype=self.dtype)
            self.logger.info('encoder_embeddings %s', self.encoder_embeddings)
            
            # encoder_inputs_embedded : [batch_size, encoder_max_time_steps, embedding_size]
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.encoder_embeddings, ids=self.encoder_inputs,
                                                                  name='inputs_embedded')
            self.logger.info('encoder_inputs_embedded %s', self.encoder_inputs_embedded)
            
            # encoder_inputs_embedded_dense: [batch_size, encoder_max_time_steps, hidden_units]
            self.encoder_inputs_embedded_dense = tf.layers.dense(inputs=self.encoder_inputs_embedded,
                                                                 units=self.hidden_units,
                                                                 use_bias=False,
                                                                 name='inputs_embedded_dense')
            self.logger.info('encoder_inputs_embedded_dense %s', self.encoder_inputs_embedded_dense)
            
            if self.use_bidirectional:
                # cell forward
                cell_fw = self.build_single_cell()
                # cell backward
                cell_bw = self.build_single_cell()
                
                bi_outputs, bi_last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                            cell_bw=cell_bw,
                                                                            inputs=self.encoder_inputs_embedded_dense,
                                                                            sequence_length=self.encoder_inputs_length,
                                                                            dtype=self.dtype)
                self.logger.info('bi_outputs %s', bi_outputs)
                self.logger.info('bi_last_state %s', bi_last_state)
                # concat bi outputs
                bi_outputs = tf.layers.dense(inputs=tf.concat(bi_outputs, axis=-1), units=self.hidden_units,
                                             use_bias=False)
                self.logger.info('bi_outputs %s', bi_outputs)
                
                if self.encoder_depth > 2:
                    upper_cell = self.build_encoder_multi_cell(self.encoder_depth - 1)
                elif self.encoder_depth == 2:
                    upper_cell = self.build_single_cell()
                else:
                    upper_cell = None
                
                self.logger.info('upper_cell %s', upper_cell)
                
                if upper_cell:
                    # encoder depth >= 2
                    upper_outputs, upper_last_state = tf.nn.dynamic_rnn(cell=upper_cell, inputs=bi_outputs,
                                                                        sequence_length=self.encoder_inputs_length,
                                                                        dtype=self.dtype)
                    self.logger.info('upper_outputs %s', upper_outputs)
                    self.logger.info('upper_last_state %s', upper_last_state)
                    
                    # encoder_outputs: [batch_size, encoder_max_time_steps, hidden_units]
                    self.encoder_outputs = upper_outputs
                    self.logger.info('encoder_outputs %s', self.encoder_outputs)
                    
                    # encoder_last_state: [batch_size, hidden_units] * encoder_depth
                    self.encoder_last_state = (bi_last_state[0],) + (
                        (upper_last_state,) if self.encoder_depth == 2 else upper_last_state)
                    self.logger.info('encoder_last_state %s', self.encoder_last_state)
                else:
                    # encoder_outputs: [batch_size, encoder_max_time_steps, hidden_units]
                    self.encoder_outputs = bi_outputs
                    self.logger.info('encoder_outputs %s', self.encoder_outputs)
                    
                    # encoder_last_state: [batch_size, hidden_units] * encoder_depth
                    self.encoder_last_state = (bi_last_state[0],)
                    self.logger.info('encoder_last_state %s', self.encoder_last_state)
            
            else:
                # encoder_cell
                self.encoder_cell = self.build_encoder_multi_cell()
                self.logger.info('encoder_cell %s', self.encoder_cell)
                # encoder_outputs: [batch_size, encoder_max_time_steps, hidden_units]
                # encoder_last_state: [batch_size, hidden_units] * encoder_depth
                self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                                                                  inputs=self.encoder_inputs_embedded_dense,
                                                                                  sequence_length=self.encoder_inputs_length,
                                                                                  dtype=self.dtype)
                self.logger.info('encoder_outputs %s', self.encoder_outputs)
                self.logger.info('encoder_last_state %s', self.encoder_last_state)
    
    def build_decoder(self):
        pass


model = Seq2SeqModel()
