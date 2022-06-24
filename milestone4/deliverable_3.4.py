import elasticsearch 
import json
import torch

from sentence_transformers import SentenceTransformer, util
from flask import Flask, Response, jsonify


# Initiate Elasticsearch
es = elasticsearch.Elasticsearch('http://127.0.0.1:9200/')

# Initiate Transformer
model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Initiate Flask app
app = Flask(__name__)

excluded_search = ['Data and graphs', 'External links', 'Further reading',
                   'Medical journals', 'See also']


def load_questions(data_loc):
    with open(data_loc) as f:
        qs = json.load(f)
    return qs


def search(index, query):
    search_query = {"query": {
                       "bool": {
                           "should": { "match": {"text": query}},
                           "must_not": {"terms" : { "section_title.keyword" : excluded_search}},
                            }
                        }
                    }
    results = index.search(index='pandemic_docs', body=search_query, size=50) 
    results = results['hits']['hits']
    return results 


def return_results(results):
    text = [x['_source']['text'] for x in results]
    article_title = [x['_source']['article_title'] for x in results]
    section_title = [x['_source']['section_title'] for x in results]
    return text, article_title, section_title


def search_similarity(query, index, model):
    result = search(index, query)
    text, article_title, section_title = return_results(result)
    query_embeddings = model.encode(query).reshape(1, -1)
    text_embeddings = model.encode(text, convert_to_numpy=True)
    search_res = util.semantic_search(query_embeddings, text_embeddings, top_k=10)[0]
    search_output = []
    for val in search_res:
        output = {}
        output['text'] = text[val['corpus_id']]
        output['title'] = article_title[val['corpus_id']]
        output['section_title'] = section_title[val['corpus_id']]
        output['score'] = val['score']
        search_output.append(output)
    return search_output


@app.route('/<query>', methods=['GET'])
def results(query, index=es, model=model):
    output = search_similarity(query, index, model)
    return jsonify(output)
    

if __name__ == '__main__':
    app.run(debug=True)