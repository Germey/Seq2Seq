import tensorflow as tf
import math
from utils.config import GO, EOS, UNK


class PointerGeneratorCoverageModel():
    def __init__(self, config, mode, logger):
        """
        init model
        :param config: config dict
        :param mode: train or inference
        :param logger: logger object
        """
        assert mode.lower() in ['train', 'inference']
        self.mode = mode.lower()
        self.logger = logger
        self.init_config(config)
        self.build_placeholders()
        self.build_encoder()
        self.build_decoder()
        self.build_optimizer()
    
    def init_config(self, config):
        """
        add config to model
        :param config: config dict
        :return: None
        """
        self.config = config
        self.hidden_units = config['hidden_units']
        self.embedding_size = config['embedding_size']
        self.encoder_max_time_steps = config['encoder_max_time_steps']
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
        self.attention_units = config['attention_units']
        self.coverage_loss_weight = config['coverage_loss_weight']
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, tf.add(self.global_epoch_step, 1))
    
    def build_placeholders(self):
        """
        init placeholders
        :return: None
        """
        self.keep_prob = tf.placeholder(self.dtype, shape=[], name='keep_prob')
        
        self.oovs_max_size = tf.placeholder(tf.int32, shape=[], name='oovs_max_size')
        
        # encoder_inputs: [batch_size, encoder_time_steps]
        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[None, self.encoder_max_time_steps],
                                             name='encoder_inputs')
        self.logger.debug('encoder_inputs %s', self.encoder_inputs)
        
        # encoder_inputs_length: [batch_size]
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=[None],
                                                    name='encoder_inputs_length')
        self.logger.debug('encoder_inputs_length %s', self.encoder_inputs_length)
        
        # encoder_inputs_extend: [batch_size, encoder_time_steps]
        self.encoder_inputs_extend = tf.placeholder(dtype=tf.int32,
                                                    shape=[None, self.encoder_max_time_steps],
                                                    name='encoder_inputs_extend')
        self.logger.debug('encoder_inputs_extend %s', self.encoder_inputs_extend)
        
        # batch_size
        self.batch_size = tf.shape(self.encoder_inputs)[0]
        self.logger.debug('batch_size %s', self.batch_size)
        
        if self.mode == 'train':
            # decoder_inputs: [batch_size, decoder_time_steps]
            self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=[None, self.decoder_max_time_steps],
                                                 name='decoder_inputs')
            self.logger.debug('decoder_inputs %s', self.decoder_inputs)
            
            # decoder_inputs_extend: [batch_size, decoder_time_steps]
            self.decoder_inputs_extend = tf.placeholder(dtype=tf.int32,
                                                        shape=[None, self.decoder_max_time_steps],
                                                        name='decoder_inputs_extend')
            self.logger.debug('decoder_inputs_extend %s', self.decoder_inputs_extend)
            
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
            
            # decoder_inputs_train: [batch_size, decoder_time_steps + 1]
            self.decoder_inputs_train = tf.concat([self.decoder_start_token, self.decoder_inputs], axis=-1)
            self.logger.debug('decoder_inputs_train %s', self.decoder_inputs_train)
            
            # decoder_inputs_train_length: [batch_size]
            self.decoder_inputs_train_length = self.decoder_inputs_length + 1
            self.logger.debug('decoder_inputs_train_length %s', self.decoder_inputs_train_length)
            
            # decoder_targets_train: [batch_size, decoder_time_steps + 1]
            self.decoder_targets_train = tf.concat([self.decoder_inputs_extend, self.decoder_end_token], axis=-1)
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
        
        with tf.variable_scope('attention'):
            
            # attention_u: [hidden_units, attention_units]
            self.attention_u = tf.get_variable(name='u', shape=[self.hidden_units, self.attention_units],
                                               initializer=tf.truncated_normal_initializer)
            self.logger.debug('attention_u %s', self.attention_u)
            
            # attention_w: [hidden_units, attention_units]
            self.attention_w = tf.get_variable(name='w', shape=[self.hidden_units, self.attention_units],
                                               initializer=tf.truncated_normal_initializer)
            self.logger.debug('attention_w %s', self.attention_w)
            
            # attention_c: [hidden_units, attention_units]
            self.attention_c = tf.get_variable(name='c', shape=[self.hidden_units, self.attention_units],
                                               initializer=tf.truncated_normal_initializer)
            self.logger.debug('attention_c %s', self.attention_c)
            
            # attention_v: [attention_units, 1]
            self.attention_v = tf.get_variable(name='v', shape=[self.attention_units, 1],
                                               initializer=tf.truncated_normal_initializer)
            self.logger.debug('attention_v %s', self.attention_v)
    
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
        :param depth: encoder depth
        :return: MultiRNNCell
        """
        depth = depth if depth else self.encoder_depth
        cells = [self.build_single_cell() for _ in range(depth)]
        return tf.nn.rnn_cell.MultiRNNCell(cells=cells)
    
    def build_decoder_cell(self, depth=None):
        """
        build decoder multi cell
        :param depth: decoder depth
        :return: MultiRNNCell
        """
        depth = depth if depth else self.decoder_depth
        cells = [self.build_single_cell() for _ in range(depth)]
        return tf.nn.rnn_cell.MultiRNNCell(cells=cells)
    
    def build_encoder(self):
        """
        build encoder
        :return: None
        """
        with tf.variable_scope('encoder') as scope:
            # encoder_embeddings: [encoder_vocab_size, embedding_size]
            self.encoder_embeddings = tf.get_variable(name='embedding',
                                                      shape=[self.encoder_vocab_size, self.embedding_size],
                                                      dtype=self.dtype,
                                                      initializer=tf.random_uniform_initializer(-math.sqrt(3),
                                                                                                math.sqrt(3),
                                                                                                dtype=self.dtype))
            self.logger.debug('encoder_embeddings %s', self.encoder_embeddings)
            
            # encoder_inputs_embedded : [batch_size, encoder_time_steps, embedding_size]
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(params=self.encoder_embeddings,
                                                                  ids=self.encoder_inputs,
                                                                  name='inputs_embedded')
            self.logger.debug('encoder_inputs_embedded %s', self.encoder_inputs_embedded)
            
            # encoder_inputs_embedded_dense: [batch_size, encoder_time_steps, hidden_units]
            self.encoder_inputs_embedded_dense = tf.layers.dense(inputs=self.encoder_inputs_embedded,
                                                                 units=self.hidden_units,
                                                                 use_bias=False,
                                                                 name='inputs_embedded_dense')
            self.logger.debug('encoder_inputs_embedded_dense %s', self.encoder_inputs_embedded_dense)
            
            if self.use_bidirectional:
                # cell forward
                cell_fw = self.build_single_cell()
                # cell backward
                cell_bw = self.build_single_cell()
                
                bi_outputs, bi_last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                            cell_bw=cell_bw,
                                                                            inputs=self.encoder_inputs_embedded_dense,
                                                                            sequence_length=self.encoder_inputs_length,
                                                                            dtype=self.dtype,
                                                                            scope=scope)
                self.logger.debug('bi_outputs %s', bi_outputs)
                self.logger.debug('bi_last_state %s', bi_last_state)
                # concat bi outputs
                bi_outputs = tf.layers.dense(inputs=tf.concat(bi_outputs, axis=-1), units=self.hidden_units,
                                             use_bias=False)
                self.logger.debug('bi_outputs %s', bi_outputs)
                
                if self.encoder_depth > 2:
                    upper_cell = self.build_encoder_cell(self.encoder_depth - 1)
                elif self.encoder_depth == 2:
                    upper_cell = self.build_single_cell()
                else:
                    upper_cell = None
                
                self.logger.debug('upper_cell %s', upper_cell)
                
                if upper_cell:
                    # encoder depth >= 2
                    upper_outputs, upper_last_state = tf.nn.dynamic_rnn(cell=upper_cell, inputs=bi_outputs,
                                                                        sequence_length=self.encoder_inputs_length,
                                                                        dtype=self.dtype,
                                                                        scope=scope)
                    self.logger.debug('upper_outputs %s', upper_outputs)
                    self.logger.debug('upper_last_state %s', upper_last_state)
                    
                    # encoder_outputs: [batch_size, encoder_time_steps, hidden_units]
                    self.encoder_outputs = upper_outputs
                    self.logger.debug('encoder_outputs %s', self.encoder_outputs)
                    
                    # encoder_last_state: [batch_size, hidden_units] * encoder_depth
                    self.encoder_last_state = (bi_last_state[0],) + (
                        (upper_last_state,) if self.encoder_depth == 2 else upper_last_state)
                    self.logger.debug('encoder_last_state %s', self.encoder_last_state)
                else:
                    # encoder_outputs: [batch_size, encoder_time_steps, hidden_units]
                    self.encoder_outputs = bi_outputs
                    self.logger.debug('encoder_outputs %s', self.encoder_outputs)
                    
                    # encoder_last_state: [batch_size, hidden_units] * encoder_depth
                    self.encoder_last_state = (bi_last_state[0],)
                    self.logger.debug('encoder_last_state %s', self.encoder_last_state)
            
            else:
                # encoder_cell
                self.encoder_cell = self.build_encoder_cell()
                self.logger.debug('encoder_cell %s', self.encoder_cell)
                # encoder_outputs: [batch_size, encoder_time_steps, hidden_units]
                # encoder_last_state: encoder_depth * [batch_size, hidden_units]
                self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                                                                  inputs=self.encoder_inputs_embedded_dense,
                                                                                  sequence_length=self.encoder_inputs_length,
                                                                                  dtype=self.dtype,
                                                                                  scope=scope)
                self.logger.debug('encoder_outputs %s', self.encoder_outputs)
                self.logger.debug('encoder_last_state %s', self.encoder_last_state)
    
    def attention(self, prev_state, encoder_outputs, coverage):
        """
        calculate attention result
        :param prev_state: prev state
        :param encoder_outputs: encoder outputs
        :return: attention result
        """
        e_i = []
        c_i = []
        # encoder_outputs: encoder_time_steps * [batch_size, hidden_units]
        # output: [batch_size, hidden_units]
        for output in encoder_outputs:
            # e_i_j: [batch_size, 1]
            e_i_j = tf.matmul(
                # tanh: [batch_size, attention_units]
                tf.tanh(
                    # prev_state: [batch_size, hidden_units]
                    # attention_w: [hidden_units, attention_units]
                    tf.matmul(prev_state, self.attention_w) +
                    # output: [batch_size, hidden_units]
                    # attention_u: [hidden_units, attention_units]
                    tf.matmul(output, self.attention_u) +
                    # coverage: [batch_size, hidden_units]
                    # attention_c: [hidden_units, attention_units]
                    tf.matmul(coverage, self.attention_c)
                ),
                # attention_v: [attention_units, 1]
                self.attention_v)
            # e_i: encoder_time_steps * [batch_size, 1]
            e_i.append(e_i_j)
        # e_i: [batch_size, encoder_time_steps]
        e_i = tf.concat(e_i, axis=1)
        
        # coverage
        coverage += tf.layers.dense(e_i, self.hidden_units, use_bias=False, name='coverage_dense')
        
        # alpha_i: [batch_size, encoder_time_steps]
        alpha_i = tf.nn.softmax(e_i, axis=-1)
        # alpha_i_split: encoder_time_steps * [batch_size, 1]
        alpha_i_split = tf.split(alpha_i, alpha_i.shape[-1], axis=-1)
        # alpha_i_split: encoder_time_steps * [batch_size, 1]
        # encoder_outputs: encoder_time_steps * [batch_size, hidden_units]
        for alpha_i_j, output in zip(alpha_i_split, encoder_outputs):
            # alpha_i_j: [batch_size, 1]
            # output: [batch_size, hidden_units]
            # c_i_j: [batch_size, hidden_units]
            c_i_j = tf.multiply(alpha_i_j, output)
            # c_i: encoder_time_steps * [batch_size, hidden_units]
            c_i.append(c_i_j)
        # c_i: [batch_size, hidden_units]
        c_i = tf.reduce_sum(c_i, axis=0)
        # c_i: [batch_size, hidden_units]
        # alpha_i: [batch_size, encoder_time_steps]
        return c_i, alpha_i, coverage
    
    def build_decoder(self):
        """
        build decoder
        :return:
        """
        with tf.variable_scope('decoder'):
            # decoder_initial_state: encoder_depth * [batch_size, hidden_units]
            self.decoder_initial_state = self.encoder_last_state
            self.logger.debug('decoder_initial_state %s', self.decoder_initial_state)
            
            self.decoder_cell = self.build_decoder_cell()
            self.logger.debug('decoder_cell %s', self.decoder_cell)
            
            # decoder_embeddings: [decoder_vocab_size, embedding_size]
            self.decoder_embeddings = tf.get_variable(name='embedding',
                                                      shape=[self.decoder_vocab_size, self.embedding_size],
                                                      dtype=self.dtype,
                                                      initializer=tf.random_uniform_initializer(-math.sqrt(3),
                                                                                                math.sqrt(3),
                                                                                                dtype=self.dtype))
            self.logger.debug('decoder_embeddings %s', self.decoder_embeddings)
            
            # encoder_outputs_unstack: encoder_time_steps * [batch_size, hidden_units]
            self.encoder_outputs_unstack = tf.unstack(self.encoder_outputs, axis=1)
            
            # state: encoder_depth * [batch_size, hidden_units]
            state = self.decoder_initial_state
            
            if self.mode == 'train':
                # decoder_inputs_embedded: [batch_size, decoder_time_steps, embedding_size]
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(params=self.decoder_embeddings,
                                                                      ids=self.decoder_inputs_train)
                self.logger.debug('decoder_inputs_embedded %s', self.decoder_inputs_embedded)
                
                # decoder_inputs_embedded_unstack: decoder_time_steps * [batch_size, embedding_size]
                self.decoder_inputs_embedded_unstack = tf.unstack(self.decoder_inputs_embedded, axis=1)
                self.logger.debug('decoder_inputs_embedded_unstack %s sample %s',
                                  len(self.decoder_inputs_embedded_unstack),
                                  self.decoder_inputs_embedded_unstack[0])
                
                self.logger.debug('encoder_outputs_unstack length %s sample %s',
                                  len(self.encoder_outputs_unstack), self.encoder_outputs_unstack[0])
                
                decoder_logits = []
                attention_distributions = []
                with tf.variable_scope('loop', reuse=tf.AUTO_REUSE):
                    coverage = tf.zeros(shape=[self.batch_size, self.hidden_units])
                    for i, inputs in enumerate(self.decoder_inputs_embedded_unstack):
                        # c_i: [batch_size, hidden_units]
                        # state: [batch_size, hidden_units]
                        # inputs: [batch_size, embedding_size]
                        c_i, alpha_i, coverage = self.attention(state[-1], encoder_outputs=self.encoder_outputs_unstack,
                                                                coverage=coverage)
                        
                        # p_gen_dense: [batch_size, 1]
                        p_gen_dense = tf.layers.dense(tf.concat([c_i, state[-1], inputs], axis=-1),
                                                      units=1,
                                                      name='p_gen_dense')
                        self.logger.debug('p_gen_dense %s', p_gen_dense)
                        # p_gen: [batch_size, 1]
                        p_gen = tf.nn.sigmoid(p_gen_dense, name='p_gen_sigmoid')
                        self.logger.debug('p_gen %s', p_gen)
                        
                        inputs = tf.concat([inputs, c_i], axis=1)
                        outputs, state = self.decoder_cell(inputs=inputs, state=state)
                        
                        # outputs_logits: [batch_size, decoder_vocab_size]
                        outputs_logits = tf.layers.dense(inputs=outputs,
                                                         units=self.decoder_vocab_size,
                                                         name='outputs_dense')
                        self.logger.debug('outputs_logits %s', outputs_logits)
                        
                        # vocab_distribution: [batch_size, decoder_vocab_size]
                        vocab_distribution = tf.nn.softmax(outputs_logits, axis=-1)
                        
                        # attention_distribution: [batch_size, encoder_inputs_length]
                        attention_distribution = alpha_i
                        
                        # final_distribution: [batch_size, decoder_vocab_size + oovs_max_size]
                        final_distribution = self.merge_distribution(p_gen, attention_distribution, vocab_distribution,
                                                                     self.oovs_max_size)
                        
                        decoder_logits.append(final_distribution)
                        attention_distributions.append(attention_distribution)
                
                # decoder_outputs: [batch_size, decoder_time_steps, hidden_units]
                self.decoder_logits = tf.stack(decoder_logits, axis=1)
                self.logger.debug('decoder_logits %s', self.decoder_logits)
                
                # decoder_depth * [batch_size, hidden_units]
                self.decoder_last_state = state
                self.logger.debug('decoder_last_state %s', self.decoder_last_state)
                
                # decoder_masks: [batch_size, reduce_max(decoder_inputs_length)]
                self.decoder_masks = tf.sequence_mask(lengths=self.decoder_inputs_train_length,
                                                      maxlen=self.decoder_max_time_steps + 1,
                                                      dtype=self.dtype,
                                                      name='masks')
                self.logger.debug('decoder_masks %s', self.decoder_masks)
                
                # # loss
                # self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits,
                #                                              targets=self.decoder_targets_train,
                #                                              weights=self.decoder_masks)
                # self.logger.debug('loss %s', self.loss)
                #
                #
                # self.predicts = tf.argmax(self.config)
                
                losses_group = []
                
                for decoder_step in range(self.decoder_max_time_steps + 1):
                    # step_final_distribution: [batch_size, decoder_vocab_size + oovs_max_size]
                    step_final_distribution = decoder_logits[decoder_step]
                    # targets: [batch_size]
                    targets = self.decoder_targets_train[:, decoder_step]
                    # indices: [batch_size, 2]
                    indices = tf.stack((tf.range(0, self.batch_size), targets), axis=1)
                    self.logger.debug('indices %s', indices)
                    # step_final_probabilities: [batch_size]
                    step_final_probabilities = tf.gather_nd(step_final_distribution, indices)
                    # step_losses: [batch_size]
                    step_losses = -tf.log(step_final_probabilities)
                    self.logger.debug('step_losses %s', step_losses)
                    # losses_group: (decoder_max_time_steps + 1) * [batch_size]
                    losses_group.append(step_losses)
                
                # decoder_masked_length: [batch_size]
                decoder_masked_length = tf.reduce_sum(self.decoder_masks, axis=1)
                self.logger.debug('decoder_masked_length %s', decoder_masked_length)
                # masked_losses: (decoder_max_time_steps + 1) * [batch_size]
                masked_losses = [v * self.decoder_masks[:, decoder_step] for decoder_step, v in enumerate(losses_group)]
                self.logger.debug('masked_losses %s', masked_losses)
                
                # merged_losses: [batch_size]
                merged_losses = sum(masked_losses) / decoder_masked_length
                self.logger.debug('merged_losses %s', merged_losses)
                
                self.generator_loss = tf.reduce_mean(merged_losses)
                self.logger.debug('generator_loss %s', self.generator_loss)
                
                # coverage_matrix: [batch_size, encoder_time_steps]
                coverage_matrix = tf.zeros_like(attention_distributions[0])
                # coverage_loss
                coverage_losses = []
                for attention_distribution in attention_distributions:
                    # coverage_loss: [batch_size]
                    coverage_loss = tf.reduce_sum(tf.minimum(attention_distribution, coverage_matrix),
                                                  [1])
                    coverage_losses.append(coverage_loss)
                    coverage_matrix += attention_distribution
                
                # decoder_masked_length: [batch_size]
                decoder_masked_length = tf.reduce_sum(self.decoder_masks, axis=1)
                self.logger.debug('decoder_masked_length %s', decoder_masked_length)
                
                # masked_losses: (decoder_max_time_steps + 1) * [batch_size]
                masked_losses = [v * self.decoder_masks[:, decoder_step] for decoder_step, v in
                                 enumerate(coverage_losses)]
                self.logger.debug('masked_losses %s', masked_losses)
                
                # merged_losses: [batch_size]
                merged_losses = sum(masked_losses) / decoder_masked_length
                self.logger.debug('merged_losses %s', merged_losses)
                
                # coverage_loss: []
                self.coverage_loss = tf.reduce_mean(merged_losses)
                self.logger.debug('coverage_loss %s', self.coverage_loss)
                
                # total loss
                self.loss = self.generator_loss + self.coverage_loss_weight * self.coverage_loss
                self.logger.debug('total loss %s', self.loss)
            
            
            else:
    
                self.decoder_scores = []
                self.decoder_probabilities = []
                self.decoder_predicts = []
    
                # decoder_initial_tokens: [batch_size]
                self.decoder_initial_tokens = tf.ones(shape=[self.batch_size], dtype=tf.int32,
                                                      name='initial_tokens') * GO
                self.logger.debug('decoder_initial_tokens %s', self.decoder_initial_tokens)
    
                # decoder_initial_tokens_embedded: [batch_size, embedding_size]
                self.decoder_initial_tokens_embedded = tf.nn.embedding_lookup(params=self.decoder_embeddings,
                                                                              ids=self.decoder_initial_tokens)
                self.logger.debug('decoder_initial_tokens_embedded %s', self.decoder_initial_tokens_embedded)
    
                inputs = self.decoder_initial_tokens_embedded
                with tf.variable_scope('loop', reuse=tf.AUTO_REUSE):
                    coverage = tf.zeros(shape=[self.batch_size, self.hidden_units])

                    for _ in range(self.decoder_max_time_steps):
                        c_i, alpha_i, coverage = self.attention(state[-1], encoder_outputs=self.encoder_outputs_unstack, coverage=coverage)
            
                        # p_gen_dense: [batch_size, 1]
                        p_gen_dense = tf.layers.dense(tf.concat([c_i, state[-1], inputs], axis=-1),
                                                      units=1,
                                                      name='p_gen_dense')
            
                        self.logger.debug('p_gen_dense %s', p_gen_dense)
                        # p_gen: [batch_size, 1]
                        p_gen = tf.nn.sigmoid(p_gen_dense, name='p_gen_sigmoid')
                        self.logger.debug('p_gen %s', p_gen)
            
                        inputs = tf.concat([inputs, c_i], axis=1)
                        outputs, state = self.decoder_cell(inputs=inputs, state=state)
            
                        # outputs_logits: [batch_size, decoder_vocab_size]
                        outputs_logits = tf.layers.dense(inputs=outputs,
                                                         units=self.decoder_vocab_size,
                                                         name='outputs_dense')
                        self.logger.debug('outputs_logits %s', outputs_logits)
            
                        # vocab_distribution: [batch_size, decoder_vocab_size]
                        vocab_distribution = tf.nn.softmax(outputs_logits, axis=-1)
            
                        # attention_distribution: [batch_size, encoder_inputs_length]
                        attention_distribution = alpha_i
            
                        # final_distribution: [batch_size, decoder_vocab_size + oovs_max_size]
                        final_distribution = self.merge_distribution(p_gen, attention_distribution, vocab_distribution,
                                                                     self.oovs_max_size)
            
                        self.logger.debug('final_distribution %s', final_distribution)
            
                        self.decoder_probabilities.append(final_distribution)
            
                        # argmax index
                        predicts = tf.argmax(final_distribution, -1)
            
                        self.logger.debug('predicts %s', predicts)
            
                        self.decoder_predicts.append(predicts)
            
                        # self.logger.debug('oov_vocabs %s', self.oovs_vocabs)
            
                        greater_index = tf.cast(tf.greater_equal(predicts, self.decoder_vocab_size), tf.int64)
            
                        self.logger.debug('greater index %s', greater_index)
            
                        indices = tf.ones([self.batch_size], tf.int64) - greater_index
            
                        self.logger.debug('indices %s', indices)
            
                        input_next = tf.multiply(predicts, indices) + \
                                     tf.multiply(greater_index, tf.ones([self.batch_size], tf.int64) * UNK)
            
                        self.logger.debug('predicts %s', predicts)
            
                        # argmax probability score
                        scores = tf.reduce_max(final_distribution, -1)
            
                        self.decoder_scores.append(scores)
            
                        # next input
                        inputs = tf.nn.embedding_lookup(params=self.decoder_embeddings,
                                                        ids=input_next)
    
                self.decoder_probabilities = tf.stack(self.decoder_probabilities, axis=1)
                self.decoder_predicts = tf.stack(self.decoder_predicts, axis=1)
                self.decoder_scores = tf.stack(self.decoder_scores, axis=1)
    
                self.logger.debug('decoder_probabilities %s', self.decoder_probabilities)
                self.logger.debug('decoder_predicts %s', self.decoder_predicts)
                self.logger.debug('decoder_scores %s', self.decoder_scores)
    
    def merge_distribution(self, p_gen, attention_distribution, vocab_distribution, oovs_max_size):
        """
        merge attention_distribution and vocab_distribution
        :param p_gen:
        :param attention_distribution:
        :param vocab_distribution:
        :param oovs_max_size:
        :return:
        """
        # attention_distribution: [batch_size, attention_length]
        attention_distribution = (1 - p_gen) * attention_distribution
        self.logger.debug('attention_distribution %s', attention_distribution)
        # attention_distribution: [batch_size, decoder_vocab_size]
        
        vocab_distribution = p_gen * vocab_distribution
        self.logger.debug('vocab_distribution %s', vocab_distribution)
        
        attention_length = tf.shape(attention_distribution)[1]
        self.logger.debug('attention_length %s', attention_length)
        
        # batch_indices: [batch_size, attention_length]
        batch_indices = tf.tile(tf.expand_dims(tf.range(0, self.batch_size), axis=1), [1, attention_length])
        self.logger.debug('batch_indices %s', batch_indices)
        
        # indices: [batch_size, attention_length, 2]
        indices = tf.stack((batch_indices, self.encoder_inputs_extend), axis=2)
        self.logger.debug('indices %s', indices)
        
        # shape_extend: [batch_size, encoder_vocab_size + oovs_max_size]
        shape_extend = [self.batch_size, self.encoder_vocab_size + oovs_max_size]
        self.logger.debug('shape_extend %s', shape_extend)
        
        # attention_distribution: [batch_size, encoder_vocab_size + oovs_max_size]
        attention_distribution = tf.scatter_nd(indices, attention_distribution, shape_extend)
        self.logger.debug('attention_distribution %s', attention_distribution)
        
        # vocab_distribution: [batch_size, encoder_vocab_size + oovs_max_size]
        vocab_distribution = tf.concat([vocab_distribution, tf.zeros(shape=[self.batch_size, oovs_max_size])], axis=-1)
        self.logger.debug('vocab_distribution %s', vocab_distribution)
        
        # final_distribution: [batch_size, encoder_vocab_size + oovs_max_size]
        final_distribution = attention_distribution + vocab_distribution
        self.logger.debug('final_distribution %s', final_distribution)
        return final_distribution
    
    def build_optimizer(self):
        """
        build optimizer
        :return: None
        """
        if self.mode == 'train':
            self.logger.info('Setting optimizer...')
            
            # trainable_verbs
            self.trainable_verbs = tf.trainable_variables()
            # self.logger.debug('trainable_verbs %s', self.trainable_verbs)
            
            if self.optimizer_type.lower() == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.logger.info('Optimizer has been set')
            
            # compute gradients
            self.gradients = tf.gradients(ys=self.loss, xs=self.trainable_verbs)
            
            # clip gradients by a given maximum_gradient_norm
            self.clip_gradients, _ = tf.clip_by_global_norm(self.gradients, self.max_gradient_norm)
            
            # train op
            self.train_op = self.optimizer.apply_gradients(zip(self.clip_gradients, self.trainable_verbs),
                                                           global_step=self.global_step)
    
    def save(self, sess, save_path, var_list=None, global_step=None):
        """
        save model to ckpt
        :param sess: session object
        :param save_path: save path
        :param var_list: variables list
        :param global_step: global step
        :return: None
        """
        saver = tf.train.Saver(var_list)
        
        # save model
        saver.save(sess=sess, save_path=save_path, global_step=global_step)
        self.logger.info('model saved at %s', save_path)
    
    def restore(self, sess, save_path, var_list=None):
        """
        restore model from ckpt
        :param sess: session object
        :param save_path: save path
        :param var_list: variables list
        :return: None
        """
        saver = tf.train.Saver(var_list)
        saver.restore(sess=sess, save_path=save_path)
        self.logger.info('model restored from %s', save_path)
    
    def train(self, sess, encoder_inputs, encoder_inputs_extend, encoder_inputs_length,
              decoder_inputs, decoder_inputs_extend, decoder_inputs_length, oovs_max_size):
        """
        train process
        :param sess: session object
        :param encoder_inputs:
        :param encoder_inputs_length:
        :param decoder_inputs:
        :param decoder_inputs_length:
        :return: None
        """
        input_feed = {
            self.encoder_inputs.name: encoder_inputs,
            self.encoder_inputs_extend.name: encoder_inputs_extend,
            self.encoder_inputs_length.name: encoder_inputs_length,
            self.decoder_inputs.name: decoder_inputs,
            self.decoder_inputs_extend.name: decoder_inputs_extend,
            self.decoder_inputs_length.name: decoder_inputs_length,
            self.oovs_max_size.name: oovs_max_size,
            self.keep_prob.name: 1 - self.dropout_rate
        }
        
        output_feed = [
            self.loss,
            self.train_op,
        ]
        outputs = sess.run(fetches=output_feed, feed_dict=input_feed)
        return outputs
    
    def eval(self, sess, encoder_inputs, encoder_inputs_extend, encoder_inputs_length,
             decoder_inputs, decoder_inputs_extend, decoder_inputs_length, oovs_max_size):
        """
        eval process
        :param sess: session object
        :param encoder_inputs:
        :param encoder_inputs_length:
        :param decoder_inputs:
        :param decoder_inputs_length:
        :return: None
        """
        input_feed = {
            self.encoder_inputs.name: encoder_inputs,
            self.encoder_inputs_extend.name: encoder_inputs_extend,
            self.encoder_inputs_length.name: encoder_inputs_length,
            self.decoder_inputs.name: decoder_inputs,
            self.decoder_inputs_extend.name: decoder_inputs_extend,
            self.decoder_inputs_length.name: decoder_inputs_length,
            self.oovs_max_size.name: oovs_max_size,
            self.keep_prob.name: 1
        }
        
        output_feed = self.loss
        
        outputs = sess.run(fetches=output_feed, feed_dict=input_feed)
        return outputs
    
    def inference(self, sess, encoder_inputs, encoder_inputs_extend, encoder_inputs_length, oovs_max_size):
        """
        inference process
        :param sess: session object
        :param encoder_inputs:
        :param encoder_inputs_length:
        :return: None
        """
        input_feed = {
            self.encoder_inputs.name: encoder_inputs,
            self.encoder_inputs_extend.name: encoder_inputs_extend,
            self.encoder_inputs_length.name: encoder_inputs_length,
            self.oovs_max_size.name: oovs_max_size,
            self.keep_prob.name: 1
        }
        
        output_feed = [
            self.decoder_predicts,
            self.decoder_scores,
        ]
        outputs = sess.run(fetches=output_feed, feed_dict=input_feed)
        return outputs
