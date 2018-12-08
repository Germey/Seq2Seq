from models import *


def get_model_class(model_class):
    class_map = {
        'seq2seq': Seq2SeqModel,
        'seq2seq_attention': Seq2SeqAttentionModel,
        'pointer_generator': PointerGeneratorModel,
        'pointer_generator_lab': PointerGeneratorLabModel,
        'pointer_generator_coverage': PointerGeneratorCoverageModel,
        'pointer_generator_coverage_limit': PointerGeneratorCoverageLimitModel,
        'pointer_generator_limit': PointerGeneratorLimitModel,
        'pointer_generator_limit_lab': PointerGeneratorLimitLabModel,
        'debug_pointer_generator_limit': DebugPointerGeneratorLimitModel,
        'debug_pointer_generator': DebugPointerGeneratorModel,
        'debug_pointer_generator_coverage': DebugPointerGeneratorCoverageModel,
    }
    assert model_class in class_map.keys()
    return class_map[model_class]
