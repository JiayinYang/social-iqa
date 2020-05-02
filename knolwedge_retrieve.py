'''
Follow this offical instruction to install Elasticsearch first
https://www.elastic.co/guide/en/elasticsearch/reference/current/getting-started-install.html

#Bulk indexing: curl -H "Content-Type: application/json" -XPOST "localhost:9200/bank/_bulk?pretty&refresh" --data-binary "@atomic_natural_sentences.jsonl"
#List all indices: curl "localhost:9200/_cat/indices?v"
#Delet index: curl -XDELETE localhost:9200/atomic

Extracted knowledge are stored in following dict format as pickle file
{
    id 1:{"context": context,
        "question": question,
        "answerA": [answerA,
                    extracted sentence 1,
                    extracted sentence 2,
                    extracted sentence 3,
                    ...
                    extracted sentence 50],
        "answerB": ...,
        "answerC": ...,
    },
    id 2: {
        ...
    },
    ...
}
'''

import requests
import json
from tqdm import tqdm
import os
import re
import nltk
import pickle


def elasticsearch_curl(uri='http://localhost:9200/', json_body='', verb='get', verbose=True):
    # pass header option for content type if request has a
    # body to avoid Content-Type error in Elasticsearch v6.0
    headers = {'Content-Type': 'application/json',}
    try:
        # make HTTP verb parameter case-insensitive by converting to lower()
        if verb.lower() == "get":
            resp = requests.get(uri, headers=headers, data=json_body.encode('utf-8'))
        elif verb.lower() == "post":
            resp = requests.post(uri, headers=headers, data=json_body.encode('utf-8'))
        elif verb.lower() == "put":
            resp = requests.put(uri, headers=headers, data=json_body.encode('utf-8'))

        # read the text object string
        try:
            resp_text = json.loads(resp.text)
        except:
            resp_text = resp.text

        # catch exceptions and print errors to terminal
    except Exception as error:
        print ('\nelasticsearch_curl() error:', error)
        resp_text = error

    # return the Python dict of the request
    if verbose: #or resp_text['errors']
        print ("resp_text:", resp_text)

    return resp_text

def fix_space_limit_bug():
    ### Use this piece if code when there is error like: "cluster_block_exception".  ###
    '''
    # This will remove limitation was set automaticlly due to lack of disk speace 
    data = '''
    {
        "transient": {
        "cluster.routing.allocation.disk.watermark.low": "100gb",
        "cluster.routing.allocation.disk.watermark.high": "50gb",
        "cluster.routing.allocation.disk.watermark.flood_stage": "10gb",
        "cluster.info.update.interval": "1m"
        }
    }
    '''
    elasticsearch_curl(uri='http://localhost:9200/_cluster/settings', json_body=data, verb='put')
    elasticsearch_curl(uri='http://localhost:9200/_all/_settings', json_body='{"index.blocks.read_only_allow_delete": null}', verb='put')
    '''

def index():
    #### Index the whole knowledge base ###
    data = ''
    with open('atomic_natural_sentences.txt','r',encoding='utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            write_line = line.strip()
            data += '{"index":{"_id":"'+str(idx)+'"}}\n' + '{"text":"'+write_line+'"}\n'
            if idx%2000 == 0:
                #print(idx)
                elasticsearch_curl(uri='http://localhost:9200/atomic/_bulk?pretty&refresh', json_body=data, verb='post', verbose=False)
                data = ''
                #if idx == 50000:
                #break
                
    elasticsearch_curl(uri='http://localhost:9200/atomic/_bulk?pretty&refresh', json_body=data, verb='post', verbose=False)
    elasticsearch_curl(uri='http://localhost:9200/_cat/indices?v', json_body='', verb='get')

def extract_knowledge():
    stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

    data_dir = "SocialIQa/"
    for file in ['dev', 'tst', 'trn']:
        newpath = os.path.join(data_dir, "socialIQa_v1.4_"+file+".jsonl")
        writer = open(os.path.join(data_dir, "socialIQa_v1.4_"+file+"_knowledge50.txt"), 'w', encoding='utf-8')
        all_result = {}
        with open(newpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for idx, line in tqdm(enumerate(lines)):
                item = json.loads(line.strip())
                context = item["context"]
                question = item["question"]
                all_result[idx] = {"context": context, "question":question}
                answers_dict = {"answerA":item["answerA"], "answerB":item["answerB"],"answerC":item["answerC"]}
                for key, ans in answers_dict.items():
                    text = ' '.join([context, question, ans])
                    query = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(text)) if not (word[0].isupper() and pos[:2]=='NN') and pos[:2] in ['NN', 'JJ', 'VB', "RB"]]
                    query = ' '.join([word.lower() for word in query if word not in stopwords])
                    #print(nltk.pos_tag(nltk.word_tokenize(text)))
                    writer.write("## "+str(idx)+' '+query +'\n')
                    all_result[idx][key] = [ans]
                    data = '{"from" : 0, "size" : 50, "query": { "match": { "text": "'+ query +'"}}}'
                    query_result = elasticsearch_curl(uri='http://localhost:9200/atomic/_search?pretty', json_body=data, verb='get', verbose=False)
                    try:
                        for hit in query_result['hits']['hits']:
                            writer.write(str(hit['_score']) +'\t'+ hit['_source']['text']+'\n')
                            all_result[idx][key].append(hit['_source']['text'])
                    except:
                        print('Failed searching for', idx, 'in', file)
                        print(text)
                        print(query_result)
        pickle.dump(all_result, open(os.path.join(data_dir, "socialIQa_v1.4_"+file+"_knowledge50.pickle"),'wb'))


if __name__ == "__main__":
    print('Starting indexing...')
    index()
    print('Starting extracting...')
    extract_knowledge()