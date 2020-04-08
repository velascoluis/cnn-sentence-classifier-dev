import pandas as pd
import os
import argparse
import json
import numpy as np
import logging
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


def prepare_data_train(num_words, train_data_path, test_data_path, column_target_value,
                       column_text_value, val_data_pct, gcp_bucket, json_tokenizer_path,
                       x_train_data_path, x_val_data_path, y_train_data_path, y_val_data_path,
                       max_sequence_lenght_path, num_classes_path, classifier_values_path):
    logging.basicConfig(level=logging.INFO)
    logging.info('Starting data preparation step ..')
    logging.info('Input data:')
    logging.info('train_data_path:{}'.format(train_data_path))
    logging.info('test_data_path:{}'.format(test_data_path))
    logging.info('column_target_value:{}'.format(column_target_value))
    logging.info('column_text_value:{}'.format(column_text_value))
    logging.info('val_data_pct:{}'.format(val_data_pct))
    logging.info('gcp_bucket:{}'.format(gcp_bucket))
    logging.info('json_tokenizer_path:{}'.format(json_tokenizer_path))
    logging.info('x_train_data_path:{}'.format(x_train_data_path))
    logging.info('x_val_data_path:{}'.format(x_val_data_path))
    logging.info('y_train_data_path:{}'.format(y_train_data_path))
    logging.info('y_val_data_path:{}'.format(y_val_data_path))
    logging.info('max_sequence_lenght_path:{}'.format(max_sequence_lenght_path))
    logging.info('num_classes_path:{}'.format(num_classes_path))
    logging.info('classifier_values:{}'.format(classifier_values_path))
    train_data_load = pd.read_csv("gs://" + gcp_bucket + "/" + train_data_path, sep=',')
    test_data_load = pd.read_csv("gs://" + gcp_bucket + "/" + test_data_path, sep=',')
    logging.info('STEP: DATA PREP (1/6) Data loaded.')
    train_data = train_data_load.dropna()
    test_data = test_data_load.dropna()
    logging.info('STEP: DATA PREP (2/6) Drop NaNs values.')
    train_data = train_data.drop_duplicates(subset=column_text_value, keep='first', inplace=False)
    test_data = test_data.drop_duplicates(subset=column_text_value, keep='first', inplace=False)
    logging.info('STEP: DATA PREP (3/6) Data deduplicated.')
    classifier_values = train_data[column_target_value].unique()
    dic = {}
    for i, class_value in enumerate(classifier_values):
        dic[class_value] = i
    labels = train_data[column_target_value].apply(lambda x: dic[x])
    val_data = train_data.sample(frac=val_data_pct, random_state=200)
    train_data = train_data.drop(val_data.index)
    texts = train_data[column_text_value]
    logging.info('STEP: DATA PREP (4/6) Data sampled.')
    tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'', lower=True)
    tokenizer.fit_on_texts(texts)
    sequences_train = tokenizer.texts_to_sequences(texts)
    sequences_valid = tokenizer.texts_to_sequences(val_data[column_text_value])
    logging.info('STEP: DATA PREP (5/6) Tokenizer created and text fitted.')
    x_train = pad_sequences(sequences_train)
    x_val = pad_sequences(sequences_valid, maxlen=x_train.shape[1])
    y_train = to_categorical(np.asarray(labels[train_data.index]))
    y_val = to_categorical(np.asarray(labels[val_data.index]))
    logging.info('STEP: DATA PREP (6/6) Training and validation datasets generated.')

    if not os.path.exists(os.path.dirname(json_tokenizer_path)):
        os.makedirs(os.path.dirname(json_tokenizer_path))
    if not os.path.exists(os.path.dirname(x_train_data_path)):
        os.makedirs(os.path.dirname(x_train_data_path))
    if not os.path.exists(os.path.dirname(x_val_data_path)):
        os.makedirs(os.path.dirname(x_val_data_path))
    if not os.path.exists(os.path.dirname(y_train_data_path)):
        os.makedirs(os.path.dirname(y_train_data_path))
    if not os.path.exists(os.path.dirname(y_val_data_path)):
        os.makedirs(os.path.dirname(y_val_data_path))
    if not os.path.exists(os.path.dirname(max_sequence_lenght_path)):
        os.makedirs(os.path.dirname(max_sequence_lenght_path))
    if not os.path.exists(os.path.dirname(num_classes_path)):
        os.makedirs(os.path.dirname(num_classes_path))
    if not os.path.exists(os.path.dirname(classifier_values_path)):
        os.makedirs(os.path.dirname(classifier_values_path))

    logging.info('Writing output data.')
    tokenizer_json = tokenizer.to_json()
    with open(json_tokenizer_path, 'w') as f:
        f.write(json.dumps(tokenizer_json))
    with open(x_train_data_path, 'w') as f:
        f.write(json.dumps(x_train.tolist()))
    with open(x_val_data_path, 'w') as f:
        f.write(json.dumps(x_val.tolist()))
    with open(y_train_data_path, 'w') as f:
        f.write(json.dumps(y_train.tolist()))
    with open(y_val_data_path, 'w') as f:
        f.write(json.dumps(y_val.tolist()))
    with open(max_sequence_lenght_path, 'w') as f:
        f.write(json.dumps(x_train.shape[1]))
    with open(num_classes_path, 'w') as f:
        f.write(json.dumps(i + 1))
    with open(classifier_values_path, 'w') as f:
        f.write(json.dumps(classifier_values.tolist()))
    logging.info('Data preparation step finished.')


def main(params):
    prepare_data_train(params.num_words, params.train_data_path,
                       params.test_data_path, params.column_target_value,
                       params.column_text_value, params.val_data_pct, params.gcp_bucket, params.json_tokenizer_path,
                       params.x_train_data_path, params.x_val_data_path, params.y_train_data_path,
                       params.y_val_data_path,
                       params.max_sequence_lenght_path, params.num_classes_path, params.classifier_values_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='00. Data preparation step')
    parser.add_argument('--train_data_path', type=str, default='None')
    parser.add_argument('--test_data_path', type=str, default='None')
    parser.add_argument('--column_target_value', type=str, default='None')
    parser.add_argument('--column_text_value', type=str, default='None')
    parser.add_argument('--val_data_pct', type=float, default='0.2')
    parser.add_argument('--num_words', type=int, default='20000')
    parser.add_argument('--gcp_bucket', type=str, default='None')
    parser.add_argument('--json_tokenizer_path', type=str, default='None')
    parser.add_argument('--x_train_data_path', type=str, default='None')
    parser.add_argument('--x_val_data_path', type=str, default='None')
    parser.add_argument('--y_train_data_path', type=str, default='None')
    parser.add_argument('--y_val_data_path', type=str, default='None')
    parser.add_argument('--max_sequence_lenght_path', type=str, default='None')
    parser.add_argument('--num_classes_path', type=str, default='None')
    parser.add_argument('--classifier_values_path', type=str, default='None')
    params = parser.parse_args()
    main(params)
