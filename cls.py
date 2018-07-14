from models import *

def get_model_class(model_class):
    class_map = {
        'seq2seq': Seq2SeqModel,
        'seq2seq_attention': Seq2SeqAttentionModel,
        'pointer_generator': PointerGeneratorModel,
        'pointer_generator_coverage': PointerGeneratorCoverageModel,
        'debug_pointer_generator': DebugPointerGeneratorModel,
        'debug_pointer_generator_coverage': DebugPointerGeneratorCoverageModel,
    }
    assert model_class in class_map.keys()
    return class_map[model_class]