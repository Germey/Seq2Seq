import tensorflow as tf

from utils.iterator import UniTextIterator, BiTextIterator

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('source_vocabulary', 'dataset/lcsts/split/vocabs.json', 'Path to source vocabulary')
    tf.app.flags.DEFINE_string('target_vocabulary', 'dataset/lcsts/split/vocabs.json', 'Path to target vocabulary')
    tf.app.flags.DEFINE_string('source_train_data', 'dataset/lcsts/split/sources.sample.txt',
                               'Path to source training data')
    tf.app.flags.DEFINE_string('target_train_data', 'dataset/lcsts/split/summaries.sample.txt',
                               'Path to target training data')
    tf.app.flags.DEFINE_string('source_valid_data', 'dataset/lcsts/split/valid.x.txt',
                               'Path to source validation data')
    tf.app.flags.DEFINE_string('target_valid_data', 'dataset/lcsts/split/valid.y.txt',
                               'Path to target validation data')
    tf.app.flags.DEFINE_integer('batch_size', 5, 'Batch size')
    
    tf.app.flags.DEFINE_integer('encoder_vocab_size', 34653, 'Source vocabulary size')
    tf.app.flags.DEFINE_integer('decoder_vocab_size', 34653, 'Target vocabulary size')
    
    tf.app.flags.DEFINE_string('split_sign', ' ', 'Separator of dataset')
    
    FLAGS = tf.app.flags.FLAGS
    
    train_set = BiTextIterator(source=FLAGS.source_train_data,
                               target=FLAGS.target_train_data,
                               source_dict=FLAGS.source_vocabulary,
                               target_dict=FLAGS.target_vocabulary,
                               batch_size=FLAGS.batch_size,
                               n_words_source=FLAGS.encoder_vocab_size,
                               n_words_target=FLAGS.decoder_vocab_size,
                               split_sign=FLAGS.split_sign,
                               )
    train_set.reset()
    
    for batch in train_set.next(extend=True):
        source_batch, target_batch, source_extend_batch, target_extend_batch, oovs_max_size, oovs_vocabs = batch
        # print('Source Batch', source_batch)
        # print('Source Extend Batch', source_extend_batch)
        # print('Target Batch', target_batch)
        print('Target Batch Extend', target_extend_batch)
        # print('Oovs Max Size', oovs_max_size)
        # print('Oovs Vocabs', oovs_vocabs)
    
    print('=' * 20)
    
    train_set.reset()
    
    # for batch in train_set.next():
    #     source_batch, target_batch = batch
    #     print('Source Batch', source_batch)
    #     print('Target Batch', target_batch)
    #
    # print('=' * 20)
    
    for batch in train_set.next(extend=True, split=True):
        source_batch, target_batch, source_extend_batch, target_extend_batch, oovs_max_size, oovs_vocabs = batch
        # print('Source Batch', source_batch)
        # print('Source Extend Batch', source_extend_batch)
        # print('Target Batch', target_batch)
        print('Target Batch Extend', target_extend_batch)
        # print('Oovs Max Size', oovs_max_size)
        # print('Oovs Vocabs', oovs_vocabs)
    
    
    #
    # train_set = UniTextIterator(source=FLAGS.source_train_data,
    #                             source_dict=FLAGS.source_vocabulary,
    #                             batch_size=FLAGS.batch_size,
    #                             n_words_source=FLAGS.encoder_vocab_size,
    #                             split_sign=FLAGS.split_sign,
    #                             )
    # train_set.reset()
    #
    # for batch in train_set.next(extend=True):
    #     source_batch, source_extend_batch, oovs_max_size, oovs_vocabs = batch
    #     print('Source Batch', source_batch)
    #     print('Source Extend Batch', source_extend_batch)
    #     print('Oovs Max Size', oovs_max_size)
    #     print('Oovs Vocabs', oovs_vocabs)
    #
    # print('=' * 20)
    #
    # train_set.reset()
    #
    # for batch in train_set.next():
    #     source_batch = batch
    #     print('Source Batch', source_batch)
    #
    # print('=' * 20)
