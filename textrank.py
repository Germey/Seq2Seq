from summa import summarizer
from textrank4zh import TextRank4Sentence

text = '日前 ， 方舟子 发文 直指 林志颖 旗下 爱碧丽 推销 假 保健品 ， 引起 哗然 。 调查 发现 ， 爱碧丽 没有 自己 的 生产 加工厂 。 其 胶原蛋白 饮品 无 核心 研发 ， 全部 代工 生产 。 号称 有 “ 逆 生长 ” 功效 的 爱碧丽 “ 梦幻 奇迹 限量 组 ” 售价 高达 1080 元 ， 实际成本 仅为 每瓶 4 元 ！'
# text = """Automatic summarization is the process of reducing a text document with a \
# computer program in order to create a summary that retains the most important points \
# of the original document. As the problem of information overload has grown, and as \
# the quantity of data has increased, so has interest in automatic summarization. \
# Technologies that can make a coherent summary take into account variables such as \
# length, writing style and syntax. An example of the use of summarization technology \
# is search engines such as Google. Document summarization is another."""

# result = summarizer.summarize(text, language='chinese')
# print(result)

tr4s = TextRank4Sentence()
tr4s.analyze(text=text, lower=True, source='all_filters')

print()
print('摘要：')
for item in tr4s.get_key_sentences(num=1):
    print(item.index, item.weight, item.sentence)  # index是语句在文本中位置，weight是权重

f = open('./dataset/lcsts/split/sources.test.txt', encoding='utf-8')
f2 = open('./dataset/lcsts/split/summaries.test.textrank.txt', 'w', encoding='utf-8')

for line in f.readlines():
    text = line.strip()
    tr4s.analyze(text=text, lower=True, source='all_filters')
    for item in tr4s.get_key_sentences(num=1):
        s = item.sentence
        print(s)
        f2.write(s.strip() + '\n')
