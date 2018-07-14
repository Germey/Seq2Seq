from rouge import Rouge
from os.path import join
import json

# config
base_path = './scores/lcsts_word_seq2seq/word'
hypothesis_file = 'summaries.inference.txt'
reference_file = 'summaries.test.txt'


def load_file(file):
    """
    load files
    :param file:
    :return:
    """
    with open(join(base_path, file), encoding='utf-8') as f:
        results = f.read().split('\n')
        results = list(filter(lambda x: x, results))
        print(len(results))
        return results

# load files
hypothesis = load_file(hypothesis_file)
reference = load_file(reference_file)

# init rouge object
rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference, avg=True)

# output
print(json.dumps(scores, indent=2))
with open(join(base_path, 'score.json'), 'w', encoding='utf-8') as f:
    f.write(json.dumps(scores, indent=2))
