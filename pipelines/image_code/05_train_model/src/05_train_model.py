import argparse
import json
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
import os
import logging

def train_model(keras_model_path, x_train_path, x_val_path, y_train_path, y_val_path,
                epochs, batch_size,
                #outputs
                output_trained_model_path):

    logging.basicConfig(level=logging.INFO)
    logging.info('Starting train model step ..')
    logging.info('Input data ..')
    logging.info('keras_model_path:{}'.format(keras_model_path))
    logging.info('x_train_path:{}'.format(x_train_path))
    logging.info('x_val_path:{}'.format(x_val_path))
    logging.info('y_train_path:{}'.format(y_train_path))
    logging.info('y_val_path:{}'.format(y_val_path))
    logging.info('epochs:{}'.format(epochs))
    logging.info('batch_size:{}'.format(batch_size))
    logging.info('output_trained_model_path:{}'.format(output_trained_model_path))

    with open(x_train_path) as f:
        x_train_json = json.load(f)
        x_train = np.array(x_train_json)
    with open(y_train_path) as f:
        y_train_json = json.load(f)
        y_train = np.array(y_train_json)
    with open(x_val_path) as f:
        x_val_json = json.load(f)
        x_val = np.array(x_val_json)
    with open(y_val_path) as f:
        y_val_json = json.load(f)
        y_val = np.array(y_val_json)
        #Load model
    model_file = open(keras_model_path, 'r')
    model_json = model_file.read()
    model_file.close()
    model = model_from_json(model_json)
    logging.info('STEP: TRAIN MODEL (1/3) Data loaded.')
    adam = Adam(lr=1e-3)
    loss = 'categorical_crossentropy'
    metrics = ['acc']
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=metrics)
    logging.info('STEP: TRAIN MODEL (2/3) Model compiled.')
    callback_earlystopping = EarlyStopping(monitor='val_loss')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val),
             callbacks=[callback_earlystopping])
    loss, acc = model.evaluate(x_train, y_train, verbose=2)
    print("Model accuracy: {:5.2f}%".format(100 * acc))
    logging.info('STEP: TRAIN MODEL (3/3) Model train finished.')
    model_output_path = output_trained_model_path
    if not os.path.exists(os.path.dirname(model_output_path)):
        os.makedirs(os.path.dirname(model_output_path))
    model.save(model_output_path, save_format='tf')
    logging.info('Train model step finished.')

def main(params):
    train_model(params.keras_model_path, params.x_train_path, params.x_val_path, params.y_train_path, params.y_val_path,
                params.batch_size, params.epochs, params.output_trained_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='05. Train model')
    parser.add_argument('--keras_model_path', type=str, default='None')
    parser.add_argument('--x_train_path', type=str, default='None')
    parser.add_argument('--x_val_path', type=str, default='None')
    parser.add_argument('--y_train_path', type=str, default='None')
    parser.add_argument('--y_val_path', type=str, default='None')
    parser.add_argument('--epochs', type=int, default='2')
    parser.add_argument('--batch_size', type=int, default='10000')
    parser.add_argument('--output_trained_model_path', type=str, default='None')
    params = parser.parse_args()
    main(params)








