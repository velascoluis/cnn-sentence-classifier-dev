# pipelines to-do
# Logs metrics - integrate w TB
# ContainerOp para el paso de train - integración con TFOperator
# Open bug capital letters - input/output
#Cambiar los print por loggers
import argparse
from typing import NamedTuple
import kfp
import kfp.compiler as compiler
import kfp.components as comp
import kfp.dsl as dsl
from kfp.components import InputPath, OutputPath


# output_emb_matrix_path: OutputPath(str)

def prepare_data_train(num_words: int, train_data_path: str, test_data_path: str, column_target_value: str,
                       column_text_value: str, val_data_pct: float, gcp_bucket: str, x_train_data_path: OutputPath(str),
                       x_val_data_path: OutputPath(str), y_train_data_path: OutputPath(str),
                       y_val_data_path: OutputPath(str), json_tokenizer_path: OutputPath(str),
                       ) -> NamedTuple('PrepareDataOutput', [('max_sequence_length', int), ('num_classes', int),
                                                             ('classifier_values', list)]):

    # Packages must be imported within the function
    import pandas as pd
    import io
    import json
    import numpy as np
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical
    from collections import namedtuple
    # Load data and delete NaN

    train_data_load = pd.read_csv("gs://" + gcp_bucket + "/" + train_data_path, sep=',')
    test_data_load = pd.read_csv("gs://" + gcp_bucket + "/" + test_data_path, sep=',')
    train_data = train_data_load.dropna()
    test_data = test_data_load.dropna()
    # Deduplication
    train_data = train_data.drop_duplicates(subset=column_text_value, keep='first', inplace=False)
    test_data = test_data.drop_duplicates(subset=column_text_value, keep='first', inplace=False)

    classifier_values = train_data[column_target_value].unique()
    dic = {}
    for i, class_value in enumerate(classifier_values):
        dic[class_value] = i
    labels = train_data[column_target_value].apply(lambda x: dic[x])
    val_data = train_data.sample(frac=val_data_pct, random_state=200)
    train_data = train_data.drop(val_data.index)
    texts = train_data[column_text_value]
    tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'', lower=True)
    tokenizer.fit_on_texts(texts)

    # serialize tokens in json file

    # storage_client = storage.Client()
    # bucket = storage_client.bucket(gcp_bucket)
    # blob = bucket.blob(json_tokenizer_path)

    # json_object = json.dumps(tokenizer_json, ensure_ascii=False)
    # blob.upload_from_string(
    #    data=json_object,
    #    content_type='application/json'
    # )

    sequences_train = tokenizer.texts_to_sequences(texts)
    sequences_valid = tokenizer.texts_to_sequences(val_data[column_text_value])
    x_train = pad_sequences(sequences_train)
    x_val = pad_sequences(sequences_valid, maxlen=x_train.shape[1])
    y_train = to_categorical(np.asarray(labels[train_data.index]))
    y_val = to_categorical(np.asarray(labels[val_data.index]))

    tokenizer_json = tokenizer.to_json()
    with io.open(json_tokenizer_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_json, f)

    with io.open(x_train_data_path, 'w', encoding='utf-8') as f:
        json.dump(x_train.tolist(), f)

    with io.open(x_val_data_path, 'w', encoding='utf-8') as f:
        json.dump(x_val.tolist(), f)

    with io.open(y_train_data_path, 'w', encoding='utf-8') as f:
        json.dump(y_train.tolist(), f)

    with io.open(y_val_data_path, 'w', encoding='utf-8') as f:
        json.dump(y_val.tolist(), f)

    PrepareDataOutput = namedtuple('PrepareDataOuput',
                                   ['max_sequence_length', 'num_classes',
                                    'classifier_values'])
    return PrepareDataOutput(x_train.shape[1],
                             i + 1, classifier_values.tolist())


def prepare_embeddings(gcp_bucket: str, num_words: int, w2v_model_path: str, embedding_dim: int,
                       json_tokenizer_path: InputPath(str), num_classes: int,
                       output_emb_matrix_path: OutputPath(str)) -> NamedTuple('PrepareEmbOutput',
                                                                              [('vocabulary_size', int)]):
    from gensim.models import Word2Vec
    from google.cloud import storage
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    import os
    import json
    import numpy as np
    from collections import namedtuple
    # Storage client for loading the w2v model
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcp_bucket)
    # Load w2v model
    model = Word2Vec()
    blob_w2v = bucket.get_blob(w2v_model_path)
    destination_uri = '{}/{}'.format(".", blob_w2v.name)
    if not os.path.exists(destination_uri):
        os.mkdir("/model")
    blob_w2v.download_to_filename(destination_uri)
    w2v_model = model.wv.load(destination_uri)
    word_vectors = w2v_model.wv

    # Load Json tokenizer

    # blob_tok = bucket.get_blob(json_tokenizer_path)
    with open(json_tokenizer_path) as f:
        json_token = json.load(f)

    tokenizer = tokenizer_from_json(json_token)
    word_index = tokenizer.word_index

    vocabulary_size = min(len(word_index) + 1, num_words)
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
    # Save the matrix @ output_embd_matrix_path
    embedding_matrix.tofile(output_emb_matrix_path)

    PrepareEmbOutput = namedtuple('PrepareEmbOuput',
                                  ['vocabulary_size'])
    return (PrepareEmbOutput(vocabulary_size))


def generate_model(drop: float, filter_sizes: list, num_filters: int, embedding_dim: int,
                   input_emb_matrix_path: InputPath(str), vocabulary_size: int, sequence_length: int, num_classes: int,
                   keras_model_path: OutputPath(str)):
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout, concatenate, Reshape, \
        Flatten
    from tensorflow.keras.models import Model
    import numpy as np

    embedding_matrix_load = np.fromfile(input_emb_matrix_path, dtype=np.int32)
    embedding_matrix = embedding_matrix_load.reshape((vocabulary_size, embedding_dim))
    embedding_layer = Embedding(vocabulary_size, embedding_dim, weights=[embedding_matrix], trainable=True)
    inputs = Input(shape=(sequence_length,))
    embedding = embedding_layer(inputs)
    reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)
    conv_0 = Conv2D(num_filters, (filter_sizes[0], embedding_dim), activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_1 = Conv2D(num_filters, (filter_sizes[1], embedding_dim), activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_2 = Conv2D(num_filters, (filter_sizes[2], embedding_dim), activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(reshape)
    maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1))(conv_0)
    maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1))(conv_1)
    maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1))(conv_2)
    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
    flatten = Flatten()(merged_tensor)
    reshape = Reshape((3 * num_filters,))(flatten)
    dropout = Dropout(drop)(flatten)
    # units -- output classes
    output = Dense(units=num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(dropout)
    model = Model(inputs, output)
    print(model.summary())
    model_json = model.to_json()
    with open(keras_model_path,'w') as f:
        f.write(model_json)
        f.close()
    print('Model generated')

def train_model(keras_model_path: InputPath(str), x_train_path: InputPath(str), x_val_path: InputPath(str), y_train_path: InputPath(str), y_val_path: InputPath(str),
                batch_size: int, epochs: int, model_output_path: OutputPath(str)):
    import json
    import numpy as np
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.models import model_from_json
    from tensorflow.keras.optimizers import Adam

    #Load data

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

    adam = Adam(lr=1e-3)
    loss = 'categorical_crossentropy'
    metrics = ['acc']

    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=metrics)

    #tensorboard metrics

    #now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    #root_logdir = "model/tf_logs"
    #if not os.path.exists(root_logdir):
    #    os.mkdir(root_logdir)
    #log_dir = "{}/run-{}/".format(root_logdir, now)
    #callback_tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    #model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val),
    #         callbacks=[callback_earlystopping, callback_tensorboard])
    callback_earlystopping = EarlyStopping(monitor='val_loss')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val),
             callbacks=[callback_earlystopping])

    loss, acc = model.evaluate(x_train, y_train, verbose=2)
    print("Model accuracy: {:5.2f}%".format(100 * acc))
    model.save(model_output_path, save_format='tf')


def deploy_model(namespace: str, trained_model_path: InputPath(str)):
    from kubernetes import client
    from kfserving import KFServingClient
    from kfserving import constants
    from kfserving import V1alpha2EndpointSpec
    from kfserving import V1alpha2PredictorSpec
    from kfserving import V1alpha2TensorflowSpec
    from kfserving import V1alpha2InferenceServiceSpec
    from kfserving import V1alpha2InferenceService
    from kubernetes.client import V1ResourceRequirements

    api_version = constants.KFSERVING_GROUP + '/' + constants.KFSERVING_VERSION
    inference_service_name = 'inference112cbk'
    default_endpoint_spec = V1alpha2EndpointSpec(
        predictor=V1alpha2PredictorSpec(
            tensorflow=V1alpha2TensorflowSpec(
                storage_uri=trained_model_path,
                resources=V1ResourceRequirements(
                    requests={'cpu': '100m', 'memory': '1Gi'},
                    limits={'cpu': '100m', 'memory': '1Gi'}))))

    isvc = V1alpha2InferenceService(api_version=api_version,
                                    kind=constants.KFSERVING_KIND,
                                    metadata=client.V1ObjectMeta(
                                        name=inference_service_name, namespace=namespace),
                                    spec=V1alpha2InferenceServiceSpec(default=default_endpoint_spec))

    KFServing = KFServingClient()
    KFServing.create(isvc)
    print('Inference service ' + inference_service_name + " created ...")
    KFServing.get(inference_service_name, namespace=namespace, watch=True, timeout_seconds=120)
    print('Model deployed')


def main(params):
    print('Generating pipeline ...')
    # Define operations
    prepare_data_train_operation = comp.func_to_container_op(prepare_data_train,
                                                             packages_to_install=['keras==2.3.1', 'pandas==1.0.1',
                                                                                  'tensorflow==2.1.0', 'numpy==1.18.1',
                                                                                  'gcsfs==0.6.0',
                                                                                  'google-cloud-storage==1.26.0'])
    prepare_emb_operation = comp.func_to_container_op(prepare_embeddings,
                                                      packages_to_install=['gensim==3.2.0', 'keras==2.3.1',
                                                                           'pandas==1.0.1', 'tensorflow==2.1.0',
                                                                           'numpy==1.18.1', 'gcsfs==0.6.0',
                                                                           'google-cloud-storage==1.26.0'])

    generate_model_operation = comp.func_to_container_op(generate_model,
                                                         packages_to_install=['gensim==3.2.0', 'keras==2.3.1',
                                                                              'pandas==1.0.1', 'tensorflow==2.1.0',
                                                                              'numpy==1.18.1', 'gcsfs==0.6.0',
                                                                              'google-cloud-storage==1.26.0'])
    train_model_operation = comp.func_to_container_op(train_model, packages_to_install=['gensim==3.2.0', 'keras==2.3.1',
                                                                                        'pandas==1.0.1',
                                                                                        'tensorflow==2.1.0',
                                                                                        'numpy==1.18.1', 'gcsfs==0.6.0',
                                                                                        'google-cloud-storage==1.26.0'])

    deploy_model_operation = comp.func_to_container_op(deploy_model, packages_to_install=['gensim==3.2.0', 'keras==2.3.1',
                                                                                        'pandas==1.0.1', 'kfserving==0.3.0.1',
                                                                                        'tensorflow==2.1.0',
                                                                                        'numpy==1.18.1', 'gcsfs==0.6.0',
                                                                                        'google-cloud-storage==1.26.0'])

    # Define pipeline
    @dsl.pipeline(
        name='CNN Text classifier pipeline',
        description='Pipeline for training and deploying a CNN text classifier using w2v pretrained embeddings.'
    )
    def cnn_pipeline(
            # Parameters of the pipeline, we set default values
            train_data_path=params.train_data_path,
            test_data_path=params.test_data_path,
            column_target_value=params.column_target_value,
            column_text_value=params.column_text_value,
            val_data_pct=params.val_data_pct,
            gcp_bucket=params.gcp_bucket,
            w2v_model_path=params.w2v_model_path,
            embedding_dim=params.embedding_dim,
            dropout=params.dropout,
            filter_sizes=params.filter_sizes,
            num_filters=params.num_filters,
            batch_size=params.batch_size,
            epochs=params.epochs,
            num_words=params.num_words,
            namespace=params.namespace
            ,
    ):
        # Tasks of the pipeline
        prepare_data_task = prepare_data_train_operation(num_words, train_data_path, test_data_path,
                                                         column_target_value,
                                                         column_text_value, val_data_pct,
                                                         gcp_bucket)

        prepare_emb_task = prepare_emb_operation(gcp_bucket, num_words, w2v_model_path, embedding_dim,
                                                 prepare_data_task.outputs['json_tokenizer'],
                                                 prepare_data_task.outputs['num_classes'])

        # Important note: ouput paremeter file is not output_emb_matrix_path, the _path is trimmed! to make it more natural

        generate_model_task =  generate_model_operation(dropout,filter_sizes,num_filters,embedding_dim,prepare_emb_task.outputs['output_emb_matrix'],
                                                        prepare_emb_task.outputs['vocabulary_size'],prepare_data_task.outputs['max_sequence_length'],
                                                        prepare_data_task.outputs['num_classes'])

        train_model_task = train_model_operation(generate_model_task.outputs['keras_model'],prepare_data_task.outputs['x_train_data'],
                                                 prepare_data_task.outputs['x_val_data'],prepare_data_task.outputs['y_train_data'],
                                                 prepare_data_task.outputs['y_val_data'],batch_size,epochs)

        deploy_model_task = deploy_model_operation(namespace,train_model_task.outputs['model_output'])

        # -----
        # Pipeline end

    # Generate .zip file
    pipeline_func = cnn_pipeline
    pipeline_filename = pipeline_func.__name__ + '.kf_pipeline.zip'
    compiler.Compiler().compile(pipeline_func, pipeline_filename)

    # Launch a test experiment
    # Define the client
    host = params.host
    client_id = params.client_id
    other_client_id = params.other_client_id
    other_client_secret = params.other_client_secret
    namespace = params.namespace

    client = kfp.Client(host=host, client_id=client_id, namespace=namespace, other_client_id=other_client_id,
                        other_client_secret=other_client_secret)
    # Experiment
    experiment = client.create_experiment('dev-testing-lightweight')
    experiment_name = 'dev-testing-lightweight'
    arguments = {"train_data_path": params.train_data_path,
                 "test_data_path": params.test_data_path,
                 "column_target_value": params.column_target_value,
                 "column_text_value": params.column_text_value,
                 "val_data_pct": params.val_data_pct,
                 "gcp_bucket": params.gcp_bucket,
                 "num_words": params.num_words,
                 "w2v_model_path": params.w2v_model_path,
                 "batch_size": params.batch_size,
                 "epochs": params.epochs,
                 "embedding_dim": params.embedding_dim,
                 "namespace": params.namespace
                 }
    run_name = pipeline_func.__name__ + ' run'
    run_result = client.run_pipeline(
        experiment_id=experiment.id,
        job_name=run_name,
        pipeline_package_path=pipeline_filename,
        params=arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lightweight build-train-deploy pipeline')
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--column_target_value', type=str)
    parser.add_argument('--column_text_value', type=str)
    parser.add_argument('--val_data_pct', type=float, default='0.2')
    parser.add_argument('--num_words', type=int, default='20000')
    parser.add_argument('--gcp_bucket', type=str)
    parser.add_argument('--json_tokenizer_path', type=str, default='model/tokens.json')
    parser.add_argument('--w2v_model_path', type=str, default='model/word2vec100d.txt')
    parser.add_argument('--embedding_dim', type=int, default='100')
    parser.add_argument('--dropout', type=float, default='0.5')
    parser.add_argument('--filter_sizes', type=list)
    parser.add_argument('--num_filters', type=int)
    parser.add_argument('--epochs', type=int, default='10')
    parser.add_argument('--batch_size', type=int, default='500')
    parser.add_argument('--host', type=str)
    parser.add_argument('--client_id', type=str)
    parser.add_argument('--other_client_id')
    parser.add_argument('--other_client_secret')
    parser.add_argument('--namespace', type=str)
    params = parser.parse_args()
    main(params)
