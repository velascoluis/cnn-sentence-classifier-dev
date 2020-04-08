import argparse
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout, concatenate, Reshape, \
    Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
import numpy as np
import os
import logging


def generate_model(num_conv_layers,maxpool_strides,drop, filter_sizes_str, num_filters, embbeding_dim,
                   input_emb_matrix_path, vocabulary_size_path, sequence_length_path, num_classes_path,
                   #outputs
                   keras_model_path):
    logging.basicConfig(level=logging.INFO)
    logging.info('Starting generate model step ..')

    logging.info('Input data ..')
    logging.info('num_conv_layers'.format(num_conv_layers))
    logging.info('maxpool_strides:{}'.format(maxpool_strides))
    logging.info('dropout'.format(drop))
    logging.info('filter_sizes:{}'.format(filter_sizes_str))
    logging.info('num_filters:{}'.format(num_filters))
    logging.info('embbeding_dim:{}'.format(embbeding_dim))
    logging.info('input_emb_matrix_path:{}'.format(input_emb_matrix_path))
    logging.info('vocabulary_size_path:{}'.format(vocabulary_size_path))
    logging.info('max_sequence_lenght_path:{}'.format(sequence_length_path))
    logging.info('num_classes_path:{}'.format(num_classes_path))
    logging.info('output_keras_model_path:{}'.format(keras_model_path))

    #Load data
    embedding_matrix_load = np.fromfile(input_emb_matrix_path, dtype=np.int32)
    with open(vocabulary_size_path) as f:
        vocabulary_size = f.readline()
    with open(sequence_length_path) as f:
        sequence_length = f.readline()
    with open(num_classes_path) as f:
        num_classes = f.readline()
    logging.info('STEP: GEN MODEL (1/2) Data loaded.')
    #workaround - double list for parameter passing, double invocation
    filter_sizes = eval(filter_sizes_str)
    strides_sizes=eval(maxpool_strides)
    embedding_matrix = embedding_matrix_load.reshape(( int(vocabulary_size), int(embbeding_dim)))
    embedding_layer = Embedding(int(vocabulary_size), embbeding_dim, weights=[embedding_matrix], trainable=True)
    input_layer = Input(shape=(int(sequence_length),))
    embedding = embedding_layer(input_layer)
    reshape_layer = Reshape(( int(sequence_length), int(embbeding_dim), 1))(embedding)
    filter_sizes = []
    for i in range(0, num_conv_layers * 2 - 1, 2):
        filter_sizes.append(i + 3)
    convolutions = []
    for layer_index in range(num_conv_layers):
        conv_layer = Conv2D(num_filters, (filter_sizes[layer_index], int(embbeding_dim)), activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape_layer)
        maxpool_layer = MaxPooling2D((int(sequence_length) - filter_sizes[layer_index] + 1, 1), strides=(1, 1))(conv_layer)
        convolutions.append(maxpool_layer)
    if (num_conv_layers > 1):
        concat_layers = concatenate(convolutions, axis=1)
    else:
        concat_layers = convolutions[0]
    flatten = Flatten()(concat_layers)
    reshape = Reshape((num_conv_layers * num_filters,))(flatten)
    dropout = Dropout(drop)(flatten)
    # units -- output classes
    output_layer = Dense(units=int(num_classes), activation='softmax', kernel_regularizer=regularizers.l2(0.01))(dropout)
    model = Model(input_layer, output_layer)
    logging.info('STEP: GEN MODEL (2/2) Model arch. defined.')
    logging.info(model.summary())
    model_json = model.to_json()
    if not os.path.exists(os.path.dirname(keras_model_path)):
        os.makedirs(os.path.dirname(keras_model_path))
    with open(keras_model_path,'w') as f:
        f.write(model_json)
    logging.info('Generate model step finished.')


def main(params):
    generate_model(params.num_conv_layers,params.maxpool_strides,params.dropout,params.filter_sizes,params.num_filters,
                       params.embbeding_dim, params.input_emb_matrix_path,params.vocabulary_size_path, params.max_sequence_lenght_path, params.num_classes_path,params.output_keras_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='03. Generate model step')
    parser.add_argument('--num_conv_layers', type=int, default=3)
    parser.add_argument('--maxpool_strides', type=str, default=[1,1])
    parser.add_argument('--dropout', type=float, default='0.5')
    parser.add_argument('--filter_sizes', type=str, default=[3, 4, 5])
    parser.add_argument('--num_filters', type=int, default='10')
    parser.add_argument('--embbeding_dim', type=int, default='100')
    parser.add_argument('--input_emb_matrix_path', type=str, default='None')
    parser.add_argument('--vocabulary_size_path', type=str, default='None')
    parser.add_argument('--max_sequence_lenght_path', type=str, default='None')
    parser.add_argument('--num_classes_path', type=str, default='None')
    parser.add_argument('--output_keras_model_path', type=str, default='None')
    params = parser.parse_args()
    main(params)











