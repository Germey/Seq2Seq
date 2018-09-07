from preprocess.writer import Writer
from preprocess.vocab import VocabTransformer
import json
from os.path import exists, join
from os import makedirs
from preprocess.pipeline import *

output_dir = join('dataset', 'bytecup', 'word')

vocab_size_limit = 30000

# pipelines enabled
if not exists(output_dir):
    makedirs(output_dir)

# pipelines and writer to process data
pipelines = [
    # StripPipeline(),
    # ReplacePipeline(),
    # JiebaPipeline(),
    # CharPipeline()
]

writer = Writer(folder=output_dir)
vocab_transformer = VocabTransformer(limit=vocab_size_limit)

train_flag, eval_flag, test_flag = True, False, False

if train_flag:
    # train
    sources_file = './dataset/bytecup/word/contents.txt'
    # summaries_file = './dataset/bytecup/titles.train.txt'
    
    sources = []
    # summaries = []
    #
    for line in open(sources_file, encoding='utf-8').readlines():
        sources.append(line.strip())
    #
    # for line in open(sources_file, encoding='utf-8').readlines():
    #     sources.append(line.strip())
    #
    # # text = open(file, encoding='utf-8').read()
    # # pattern = re.compile('<doc id=(\d+)>.*?<summary>(.*?)</summary>.*?<short_text>(.*?)</short_text>.*?</doc>', re.S)
    # # results = re.findall(pattern, text)
    #
    #
    #
    # for result in results:
    #     summary = result[1].strip()
    #     source = result[2].strip()
    #     sources.append(source)
    #     summaries.append(summary)
    #
    # # pre precess by pipeline
    # for pipeline in pipelines:
    #     print('Running', pipeline)
    #     sources = pipeline.process_all(sources)
    #     summaries = pipeline.process_all(summaries)
    print('Sources', len(sources))
    # get vocabs of articles and summaries, they use the same vocabs
    word2id, id2word = vocab_transformer.build_vocabs(sources)
    
    # write data to txt
    # writer.write_to_txt(sources, 'sources.train.txt')
    # writer.write_to_txt(summaries, 'summaries.train.txt')
    
    # write vocab to json
    writer.write_to_json(word2id, 'vocabs_lower.json')

if eval_flag:
    
    # eval
    file = './dataset/lcsts/origin/LCSTS/DATA/PART_II.txt'
    
    # start processing
    sources = []
    summaries = []
    
    text = open(file, encoding='utf-8').read()
    pattern = re.compile('<doc id=(\d+)>.*?<summary>(.*?)</summary>.*?<short_text>(.*?)</short_text>.*?</doc>', re.S)
    results = re.findall(pattern, text)
    
    for result in results:
        summary = result[1].strip()
        source = result[2].strip()
        sources.append(source)
        summaries.append(summary)
    
    # pre precess by pipeline
    for pipeline in pipelines:
        print('Running', pipeline)
        sources = pipeline.process_all(sources)
        summaries = pipeline.process_all(summaries)
    
    # write data to txt
    writer.write_to_txt(sources, 'sources.eval.txt')
    writer.write_to_txt(summaries, 'summaries.eval.txt')

if test_flag:
    # eval
    # file = './dataset/lcsts/origin/LCSTS/DATA/PART_III.txt'
    #
    # # start processing
    # sources = []
    # summaries = []
    #
    # text = open(file, encoding='utf-8').read()
    # pattern = re.compile(
    #     '<doc id=.*?>.*?<human_label>(.*?)</human_label>.*?<summary>(.*?)</summary>.*?<short_text>(.*?)</short_text>.*?</doc>',
    #     re.S)
    #
    # results = re.findall(pattern, text)
    
    # for result in results:
    #     rank = int(result[0].strip())
    #     if rank >= 2:
    #         summary = result[1].strip()
    #         source = result[2].strip()
    #         sources.append(source)
    #         summaries.append(summary)
    sources_file = '/private/var/py/Seq2Seq/dataset/lcsts/origin/LCSTS/Result/weibo.txt'
    summaries_file = '/private/var/py/Seq2Seq/dataset/lcsts/origin/LCSTS/Result/sumary.human.txt'
    sources = open(sources_file, encoding='utf-8').read().split('\n')
    summaries = open(summaries_file, encoding='utf-8').read().split('\n')
    
    # pre precess by pipeline
    for pipeline in pipelines:
        print('Running', pipeline)
        sources = pipeline.process_all(sources)
        summaries = pipeline.process_all(summaries)
    
    # write data to txt
    writer.write_to_txt(sources, 'sources.test.txt')
    writer.write_to_txt(summaries, 'summaries.test.txt')
