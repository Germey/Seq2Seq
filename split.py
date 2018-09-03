from sklearn.model_selection import train_test_split
from os.path import join

input_dir = './dataset/bytecup'
output_dir = './dataset/bytecup'

contents = []
for line in open(join(input_dir, 'contents.txt'), encoding='utf-8').readlines():
    contents.append(line.strip())

titles = []
for line in open(join(input_dir, 'titles.txt'), encoding='utf-8').readlines():
    titles.append(line.strip())

print('Contents', len(contents))
print('Titles', len(titles))

x_train, x_test, y_train, y_test = train_test_split(contents, titles, test_size=0.01)

from preprocess.writer import Writer

writer = Writer(output_dir)

writer.write_to_txt(x_train, 'contents.train.txt')
writer.write_to_txt(x_test, 'contents.eval.txt')
writer.write_to_txt(y_train, 'titles.train.txt')
writer.write_to_txt(y_test, 'titles.eval.txt')
