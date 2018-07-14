import numpy as np
import json
import pickle
import utils.config as config

extra_tokens = [config.GO, config.EOS, config.UNK]
start_token = extra_tokens.index(config.GO)  # start_token = 0
end_token = extra_tokens.index(config.EOS)  # end_token = 1
unk_token = extra_tokens.index(config.UNK)


def load_dict(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        with open(filename, 'r', encoding='utf-8') as f:
            return pickle.load(f)


class InferenceIterator():
    """Simple Text iterator."""
    
    def __init__(self, source, source_dict,
                 batch_size=128, max_length=None,
                 n_words_source=-1,
                 skip_empty=False,
                 sort_by_length=False,
                 encoding='utf-8',
                 split_sign=' '):
        
        self.source = open(source, 'r', encoding=encoding)
        self.source_dict = load_dict(source_dict)
        self.batch_size = batch_size
        self.max_length = max_length
        self.skip_empty = skip_empty
        self.n_words_source = n_words_source
        self.split_sign = split_sign
        
        if self.n_words_source > 0:
            for key, idx in self.source_dict.items():
                if idx >= self.n_words_source:
                    del self.source_dict[key]
        
        self.sort_by_length = sort_by_length
        self.source_buffer = []
        self.end_of_data = False
        self.reset()
    
    def length(self):
        self.reset()
        return len(self.source_buffer)
    
    def reset(self):
        self.source.seek(0)
        
        self.source_buffer = []
        self.target_buffer = []
        
        self.end_of_data = False
        # fill buffer, if it's empty
        if len(self.source_buffer) == 0:
            for ss in self.source.readlines():
                self.source_buffer.append(ss.strip().split(self.split_sign))
            # sort by buffer
            if self.sort_by_length:
                slen = np.array([len(s) for s in self.source_buffer])
                sidx = slen.argsort()
                sbuf = [self.source_buffer[i] for i in sidx]
                self.source_buffer = sbuf
            else:
                self.source_buffer.reverse()
    
    def next(self):
        """
        get next batch
        :return:
        """
        source = []
        # actual work here
        while not self.end_of_data:
            ss = None
            try:
                ss = self.source_buffer.pop()
            except IndexError:
                self.end_of_data = True
            if ss:
                ss = [self.source_dict[w] if w in self.source_dict
                      else unk_token for w in ss]
                if self.max_length and len(ss) > self.max_length:
                    continue
                if self.skip_empty and not ss:
                    continue
                source.append(ss)
            
            if self.end_of_data and len(source):
                yield source
                source = []
            
            if len(source) >= self.batch_size:
                yield source
                source = []


class TrainTextIterator():
    def __init__(self, source, target,
                 source_dict, target_dict,
                 batch_size=128,
                 max_length=None,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 sort_by_length=False,
                 encoding='utf-8',
                 split_sign=' '):
        
        self.source = open(source, 'r', encoding=encoding)
        self.target = open(target, 'r', encoding=encoding)
        
        self.source_dict = load_dict(source_dict)
        self.target_dict = load_dict(target_dict)
        
        self.batch_size = batch_size
        self.max_length = max_length
        self.skip_empty = skip_empty
        
        self.n_words_source = n_words_source
        self.n_words_target = n_words_target
        
        self.split_sign = split_sign
        
        if self.n_words_source > 0:
            for key, idx in self.source_dict.items():
                if idx >= self.n_words_source:
                    del self.source_dict[key]
        
        if self.n_words_target > 0:
            for key, idx in self.target_dict.items():
                if idx >= self.n_words_target:
                    del self.target_dict[key]
        
        self.sort_by_length = sort_by_length
        
        self.source_buffer = []
        self.target_buffer = []
        
        self.end_of_data = False
    
    def reset(self):
        """
        reset data, update buffer
        :return:
        """
        self.source.seek(0)
        self.target.seek(0)
        
        self.source_buffer = []
        self.target_buffer = []
        
        self.end_of_data = False
        
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'
        
        if len(self.source_buffer) == 0:
            for ss in self.source.readlines():
                self.source_buffer.append(ss.strip().split(self.split_sign))
            for tt in self.target.readlines():
                self.target_buffer.append(tt.strip().split(self.split_sign))
            
            # sort by target buffer
            if self.sort_by_length:
                tlen = np.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()
                sbuf = [self.source_buffer[i] for i in tidx]
                tbuf = [self.target_buffer[i] for i in tidx]
                self.source_buffer = sbuf
                self.target_buffer = tbuf
            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()
    
    def length(self):
        """
        get length of data
        :return:
        """
        self.reset()
        return len(self.source_buffer)
    
    def next(self):
        """
        get next batch
        :return:
        """
        source, target = [], []
        # actual work here
        while not self.end_of_data:
            ss, tt = None, None
            try:
                ss = self.source_buffer.pop()
                tt = self.target_buffer.pop()
            except IndexError:
                self.end_of_data = True
            if ss and tt:
                # transfer to dict index
                ss = [self.source_dict[w] if w in self.source_dict
                      else unk_token for w in ss]
                tt = [self.target_dict[w] if w in self.target_dict
                      else unk_token for w in tt]
                if self.max_length:
                    if len(ss) > self.max_length and len(tt) > self.max_length:
                        continue
                if self.skip_empty and (not ss or not tt):
                    continue
                
                source.append(ss)
                target.append(tt)
            
            if self.end_of_data and len(source) and len(target):
                yield source, target
                source, target = [], []
            
            if len(source) >= self.batch_size and len(target) >= self.batch_size:
                yield source, target
                source, target = [], []


class ExtendTextIterator():
    def __init__(self, source, target,
                 source_dict, target_dict,
                 batch_size=128,
                 max_length=None,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 sort_by_length=False,
                 encoding='utf-8',
                 split_sign=' '):
        
        assert source_dict == target_dict
        
        self.source = open(source, 'r', encoding=encoding)
        self.target = open(target, 'r', encoding=encoding)
        
        self.source_dict = load_dict(source_dict)
        self.target_dict = load_dict(target_dict)
        
        self.batch_size = batch_size
        self.max_length = max_length
        self.skip_empty = skip_empty
        
        self.n_words_source = n_words_source
        self.n_words_target = n_words_target
        
        self.split_sign = split_sign
        
        if self.n_words_source > 0:
            for key, idx in self.source_dict.items():
                if idx >= self.n_words_source:
                    del self.source_dict[key]
        
        if self.n_words_target > 0:
            for key, idx in self.target_dict.items():
                if idx >= self.n_words_target:
                    del self.target_dict[key]
        
        self.sort_by_length = sort_by_length
        
        self.source_buffer = []
        self.target_buffer = []
        
        self.end_of_data = False
    
    def reset(self):
        """
        reset data, update buffer
        :return:
        """
        self.source.seek(0)
        self.target.seek(0)
        
        self.source_buffer = []
        self.target_buffer = []
        
        self.end_of_data = False
        
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'
        
        if len(self.source_buffer) == 0:
            for ss in self.source.readlines():
                self.source_buffer.append(ss.strip().split(self.split_sign))
            for tt in self.target.readlines():
                self.target_buffer.append(tt.strip().split(self.split_sign))
            
            # sort by target buffer
            if self.sort_by_length:
                tlen = np.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()
                sbuf = [self.source_buffer[i] for i in tidx]
                tbuf = [self.target_buffer[i] for i in tidx]
                self.source_buffer = sbuf
                self.target_buffer = tbuf
            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()
    
    def length(self):
        """
        get length of data
        :return:
        """
        self.reset()
        return len(self.source_buffer)
    
    def extend(self, source, target):
        """
        extend vocab
        :param source:
        :param target:
        :return:
        """
        oovs = []
        oovs_vocab = {}
        source_ids_extend, target_ids_extend = [], []
        for w in source:
            if not w in self.source_dict:
                if w not in oovs:  # Add to list of OOVs
                    oovs.append(w)
                    # oovs_vocab[w] =
                oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
                source_ids_extend.append(len(
                    self.source_dict) + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
                oovs_vocab[w] = len(self.source_dict) + oov_num
            else:
                source_ids_extend.append(self.source_dict[w])
        for w in target:
            if not w in self.target_dict:  # If w is an OOV word
                if w in oovs:  # If w is an in-article OOV
                    target_ids_extend.append(len(self.target_dict) + oovs.index(w))
                else:  # If w is an out-of-article OOV
                    target_ids_extend.append(unk_token)  # Map to the UNK token id
            else:
                target_ids_extend.append(self.target_dict[w])
        return source_ids_extend, target_ids_extend, oovs_vocab
    
    def next(self):
        """
        get next batch
        :return:
        """
        source, target, source_extend, target_extend, oovs_vocabs = [], [], [], [], []
        oovs_max_size = 0
        # actual work here
        while not self.end_of_data:
            source_item, target_item = None, None
            try:
                source_item = self.source_buffer.pop()
                target_item = self.target_buffer.pop()
            except IndexError:
                self.end_of_data = True
            if source_item and target_item:
                # transfer to dict index
                source_ids = [self.source_dict[w] if w in self.source_dict
                              else unk_token for w in source_item]
                target_ids = [self.target_dict[w] if w in self.target_dict
                              else unk_token for w in target_item]
                source_ids_extend, target_ids_extend, oovs_vocab = self.extend(source_item, target_item)
                if len(oovs_vocab) > oovs_max_size:
                    oovs_max_size = len(oovs_vocab)
                if self.max_length:
                    if len(source_ids) > self.max_length and len(target_ids) > self.max_length:
                        continue
                if self.skip_empty and (not source_ids or not target_ids):
                    continue
                
                source.append(source_ids)
                target.append(target_ids)
                source_extend.append(source_ids_extend)
                target_extend.append(target_ids_extend)
                oovs_vocabs.append(oovs_vocab)
            
            if self.end_of_data and len(source) and len(target):
                yield source, target, source_extend, target_extend, oovs_max_size, oovs_vocabs
                source, target, source_extend, target_extend, oovs_vocabs = [], [], [], [], []
                oovs_max_size = 0
            
            if len(source) >= self.batch_size and len(target) >= self.batch_size:
                yield source, target, source_extend, target_extend, oovs_max_size, oovs_vocabs
                source, target, source_extend, target_extend, oovs_vocabs = [], [], [], [], []
                oovs_max_size = 0
