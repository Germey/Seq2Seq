# !/usr/bin/env python
# coding: utf-8
import os
import logging
import jieba
from preprocess.config import UNK
from utils.iterator import UniTextIterator, end_token
from utils.funcs import prepare_batch, load_inverse_dict, inverse_dict
import json
import tensorflow as tf
from cls import get_model_class

# Decoding parameters
tf.app.flags.DEFINE_integer('beam_width', 1, 'Beam width used in beam search')
tf.app.flags.DEFINE_integer('inference_batch_size', 256, 'Batch size used for decoding')
tf.app.flags.DEFINE_integer('max_inference_step', 60, 'Maximum time step limit to decode')
tf.app.flags.DEFINE_string('model_path', 'checkpoints/lcsts_split_pointer_generator/lcsts.ckpt-685000',
                           'Path to a specific model checkpoint.')
tf.app.flags.DEFINE_string('inference_input', 'storage/system.input.txt', 'Decoding input path')
tf.app.flags.DEFINE_string('inference_output', 'storage/system.output.txt', 'Decoding output path')

# Runtime parameters
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')
tf.app.flags.DEFINE_string('gpu', '0', 'GPU Number')
tf.app.flags.DEFINE_boolean('debug', True, 'Enable debug mode')
tf.app.flags.DEFINE_boolean('extend_vocabs', True, 'Extend oovs vocabs')
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
    """
    load model
    :param session: session object
    :param config: config dict
    :return:
    """
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
            if result == '<UNK>':
                words.append(result)
            else:
                words.append(result)
        else:
            words.append(UNK)
    return ' '.join(words)


# def decode():
# os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

# Load model config
config = load_config(FLAGS)
print(config)
# Load source data to decode

# Load inverse dictionary used in decoding
target_inverse_dict = load_inverse_dict(config['target_vocabulary'])

# Initiate TF session
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                        log_device_placement=FLAGS.log_device_placement,
                                        gpu_options=tf.GPUOptions(allow_growth=True)))

# Reload existing checkpoint
model = load_model(sess, config)
logger.info('Decoding %s...', FLAGS.inference_input)

fout = open(FLAGS.inference_output, 'w', encoding='utf-8')

import jieba

from flask import Flask, request, render_template

space = ' '

app = Flask(__name__)


@app.route('/summarize', methods=['GET'])
def summarize():
    source = request.args.get('source')
    source = space.join(jieba.lcut(source.replace(space, '')))
    
    with open(FLAGS.inference_input, 'w', encoding='utf-8') as f:
        f.write(source)
    print('Processing', source)
    test_set = UniTextIterator(source=config['inference_input'],
                               split_sign=config['split_sign'],
                               batch_size=config['inference_batch_size'],
                               source_dict=config['source_vocabulary'],
                               n_words_source=config['encoder_vocab_size'])
    
    test_set.reset()
    result, probabilities, p_gens, scores = None, None, None, None
    for idx, batch in enumerate(test_set.next(extend=FLAGS.extend_vocabs)):
        source_batch, source_extend_batch, oovs_max_size, oovs_vocabs = batch
        
        source, source_len = prepare_batch(source_batch, config['encoder_max_time_steps'])
        source_extend, _ = prepare_batch(source_extend_batch, config['encoder_max_time_steps'])
        
        predicts, scores, probabilities, p_gens, greater_indices = model.inference(sess,
                                                                                   encoder_inputs=source,
                                                                                   encoder_inputs_extend=source_extend,
                                                                                   encoder_inputs_length=source_len,
                                                                                   oovs_max_size=oovs_max_size)
        print('Shape', predicts.shape, scores.shape, probabilities.shape, p_gens.shape)
        for predict_seq, score_seq, prob_seq, p_gen_seq, oovs_vocab in zip(predicts, scores, probabilities,
                                                                           p_gens, oovs_vocabs):
            print('Score', score_seq, 'predict_seq', predict_seq, 'p_gen_seq', p_gen_seq)
            result = seq2words(predict_seq, inverse_target_dictionary=target_inverse_dict,
                               oovs_vocab=inverse_dict(oovs_vocab))
            logger.info('result %s', result)
            fout.write(result + '\n')
    
    logger.info('Finished!')
    return json.dumps({
        'summarization': result,
        'gens': p_gens.tolist()[0],
        # 'probabilities': probabilities.tolist()[0],
        'scores': scores.tolist()[0]
    }, ensure_ascii=False)


# if __name__ == '__main__':
#     decode()

# if __name__ == '__main__':
#     # source = '今天 有传 在 北京 某 小区 ， 一 光头 明星 因 吸毒 被捕 的 消息 。 下午 北京警方 官方 微博 发布 声明 通报情况 ， 证实 该 明星 为 房祖名 。 房祖名 伙同 另外 6 人 ， 于 17 日晚 在 北京 朝阳区 三里屯 某 小区 的 暂住地 内 吸食毒品 ， 6 人 全部 被 警方 抓获 ， 且 当事人 对 犯案 实施 供认不讳 。'
#     # source = '软妹币 在 全球 支付 货币 排名 已 由 2012 年 1 月份 的 第 20 位 攀升 至今 年 5 月份 的 第 13 位 ， 软妹币 支付 额 稳步增长 ， 市场份额 升至 0.84% 的 新高 。 此前 央行 发布 的 《 中国 货币政策 执行 报告 》 显示 ， 一季度 银行 累计 办理 跨境 贸易 软妹币 结算 业务 10039.2 亿元 ， 同比 增长 72.3% 。'
#     # source = '半月谈 网 4 月 以来 的 房价 涨势 令 本轮 调控 略显 尴尬 ， 而 在 推进 城镇化 将 释放 大量 住房 需求 的 背景 之下 ， 以往 通过 “ 行政 手腕 ” 调控 的 手段 恐将 越来越 “ 吃力不讨好 ” 。 多家 机构 认为 ， “ 去 行政化 ” 或 将 成为 未来 调控 方向 。'
#     source = '正 处于 风口浪尖 的 国内 奶粉 行业 出现 大 交易 。 蒙牛 乳业 （ 02319 . HK ） 以及 雅士利 （ 01230 . HK ） 昨日 发布公告 称 ， 蒙牛 乳业 将 斥资 81.5 亿港元 收购 雅士利 约 65.4% 股权 。 业界 称 ， 此举 有助于 蒙牛 乳业 补 上 奶粉 短板 ， 以期 重新 超越 伊利 成为 行业 领头羊 。'
#     decode(source)

#
# def hello_world():
#     return 'Hello World!'

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)
