from nltk.parse.corenlp import CoreNLPParser

tk = CoreNLPParser()


def tokenize(line):
    return ' '.join(list(tk.tokenize(line)))


def f2h(f_str):
    h_str = ''
    for uchar in f_str:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:
            inside_code -= 65248
        h_str += chr(inside_code)
    return h_str


def process_item(item):
    item = item.strip()
    item = item.lower()
    item = f2h(item)
    item = tokenize(item)
    item = item.strip()
    return item


fin_content = open('bytecup.corpus.validation_set.txt', encoding='utf-8')
# fin_title = open('titles.txt', encoding='utf-8')

fout_content = open('bytecup.corpus.validation_set.token.txt', 'w', encoding='utf-8')
# fout_title = open('titles.token.txt', 'w', encoding='utf-8')
import json
count = 0
error_count = 0
while True:
    content = fin_content.readline()
    content = json.loads(content.strip())['content']
    # title = fin_title.readline()
    if not content:
        break
    try:
        content = process_item(content)
        # title = process_item(title)
    except:
        error_count += 1
        continue
    fout_content.write(content + '\n')
    # fout_title.write(title + '\n')
    count += 1
    if count % 100 == 0:
        print('Processed', count, 'lines, errors', error_count)

fout_content.close()
# fout_title.close()
