from utils.iterator import ExtendTextIterator

set = ExtendTextIterator(source='test.txt',
                         target='test_target.txt',
                         source_dict='test_vocab.json',
                         target_dict='test_vocab.json',
                         batch_size=3,
                         max_length=None,
                         n_words_source=21548,
                         n_words_target=21548,
                         sort_by_length=False,
                         split_sign=' ',
                         )

set.reset()

for source, target, source_extend, target_extend, oovs_max_size in set.next():
    print('=' * 20)
    print('source', source)
    print('target', target)
    print('source_extend', source_extend)
    print('target_extend', target_extend)
    print('oovs_max_size', oovs_max_size)
