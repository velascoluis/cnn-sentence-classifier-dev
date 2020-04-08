from gensim.models import Word2Vec
from google.cloud import storage
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import os
import json
import numpy as np
import argparse
import logging


def prepare_embeddings(gcp_bucket, num_words, w2v_model_path, embedding_dim,
                       json_tokenizer_path,output_emb_matrix_path,vocabulary_size_path):
    logging.basicConfig(level=logging.INFO)
    logging.info('Starting embbedings preparation step ..')
    logging.info('Input data:')
    logging.info('gcp_bucket:{}'.format(gcp_bucket))
    logging.info('num_words:{}'.format(num_words))
    logging.info('w2v_model_path:{}'.format(w2v_model_path))
    logging.info('embedding_dim:{}'.format(embedding_dim))
    logging.info('json_tokenizer_path:{}'.format(json_tokenizer_path))
    logging.info('output_emb_matrix_path:{}'.format(output_emb_matrix_path))
    logging.info('vocabulary_size_path:{}'.format(vocabulary_size_path))
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcp_bucket)
    model = Word2Vec()
    blob_w2v = bucket.get_blob(w2v_model_path)
    destination_uri = "/02_w2v_model.bin"
    blob_w2v.download_to_filename(destination_uri)
    w2v_model = model.wv.load(destination_uri)
    word_vectors = w2v_model.wv
    logging.info('STEP: PREP EMB (1/3) Word2Vec model loaded.')
    # Load JSON tokenizer
    with open(json_tokenizer_path) as f:
        json_token = json.load(f)
    tokenizer = tokenizer_from_json(json_token)
    word_index = tokenizer.word_index
    vocabulary_size = min(len(word_index) + 1, num_words)
    logging.info('STEP: PREP EMB (2/3) Tokenizer loaded.')
    embedding_matrix = np.zeros((vocabulary_size, embedding_dim), dtype=np.int32)
    for word, i in word_index.items():
        if i >= num_words:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), embedding_dim)
    del (word_vectors)
    logging.info('STEP: PREP EMB (3/3) Embedding matrix generated.')
     #Save the matrix @ output_embd_matrix_path
    logging.info('Writing output data.')
    if not os.path.exists(os.path.dirname(output_emb_matrix_path)):
        os.makedirs(os.path.dirname(output_emb_matrix_path))
    if not os.path.exists(os.path.dirname(vocabulary_size_path)):
        os.makedirs(os.path.dirname(vocabulary_size_path))
    with open(output_emb_matrix_path, 'w') as f:
        embedding_matrix.tofile(output_emb_matrix_path)
    with open(vocabulary_size_path, 'w') as f:
        f.write(str(vocabulary_size))
    logging.info('Prepare embeddings step finished.')


def main(params):
    prepare_embeddings(params.gcp_bucket,params.num_words,params.w2v_model_path,
                       params.embbeding_dim, params.json_tokenizer_path,params.output_emb_matrix_path,params.vocabulary_size_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='02. Prepare embeddings step')
    parser.add_argument('--gcp_bucket', type=str, default='None')
    parser.add_argument('--num_words', type=int, default='20000')
    parser.add_argument('--w2v_model_path', type=str, default='None')
    parser.add_argument('--embbeding_dim', type=int, default='100')
    parser.add_argument('--json_tokenizer_path', type=str, default='None')
    parser.add_argument('--output_emb_matrix_path', type=str, default='None')
    parser.add_argument('--vocabulary_size_path', type=str, default='None')
    params = parser.parse_args()
    main(params)




