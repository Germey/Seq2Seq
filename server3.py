from flask import Flask, request, render_template
from flask_cors import CORS
from textrank4zh import TextRank4Sentence
import json

space = ' '

app = Flask(__name__)
CORS(app)

tr4s = TextRank4Sentence()


@app.route('/summarize', methods=['GET'])
def summarize():
    text = request.args.get('source')
    num = int(request.args.get('number'))
    
    results = []
    tr4s.analyze(text=text, lower=True, source='all_filters')
    
    print('摘要：')
    for item in tr4s.get_key_sentences(num=num):
        print(item.index, item.weight, item.sentence)
        
        result = {
            'weight': item.weight,
            'summarization': item.sentence
        }
        results.append(result)
    return json.dumps(results)
    
    # f = open('./dataset/lcsts/split/sources.test.txt', encoding='utf-8')
    # f2 = open('./dataset/lcsts/split/summaries.test.textrank.txt', 'w', encoding='utf-8')
    
    # for line in f.readlines():
    #     text = line.strip()
    #     tr4s.analyze(text=text, lower=True, source='all_filters')
    #     for item in tr4s.get_key_sentences(num=1):
    #         s = item.sentence
    #         print(s)
    #         f2.write(s.strip() + '\n')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5557, debug=True)
