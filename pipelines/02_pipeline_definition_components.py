import argparse
import kfp
import kfp.compiler as compiler
import kfp.dsl as dsl
from kfp import components
import datetime
from kubeflow.metadata import metadata
import logging

# Components definition files


metadata_logger_component_file = 'image_code/00_metadata_logger/component.yaml'
prepare_data_component_file = 'image_code/01_prepare_data/component.yaml'
prepare_embbedings_component_file = 'image_code/02_prepare_embeddings/component.yaml'
generate_model_component_file = 'image_code/03_generate_model/component.yaml'
move_data_pvc_component_file = 'image_code/04_move_data_pvc/component.yaml'
train_model_component_file = 'image_code/05_train_model/component.yaml'
train_model_component_file_dist = 'image_code/06_dist_launcher/component.yaml'
deploy_model_component_file_pvc = 'image_code/07_deploy_model/component_pvc.yaml'
deploy_model_component_file_par = 'image_code/07_deploy_model/component_par.yaml'

# Components load
metadata_logger_component = components.load_component_from_file(metadata_logger_component_file)
prepare_data_component = components.load_component_from_file(prepare_data_component_file)
prepare_embbedings_component = components.load_component_from_file(prepare_embbedings_component_file)
generate_model_component = components.load_component_from_file(generate_model_component_file)
move_data_pvc_component = components.load_component_from_file(move_data_pvc_component_file)
train_model_component = components.load_component_from_file(train_model_component_file)
train_model_component_dist = components.load_component_from_file(train_model_component_file_dist)
deploy_model_component_pvc = components.load_component_from_file(deploy_model_component_file_pvc)
deploy_model_component_par = components.load_component_from_file(deploy_model_component_file_par)


# Operations definition

def metadata_logger_operation(log_type, workspace_name, run_name, input_values):
    return metadata_logger_component(log_type, workspace_name, run_name, input_values)


def prepare_data_operation(train_data_path, test_data_path, column_target_value,
                           column_text_value, val_data_pct, gcp_bucket):
    return prepare_data_component(train_data_path, test_data_path,
                                  column_target_value, column_text_value, val_data_pct, gcp_bucket)


def prepare_emb_operation(gcp_bucket, num_words, w2v_model_path, embbeding_dim,
                          json_tokenizer_path):
    return prepare_embbedings_component(gcp_bucket, num_words, w2v_model_path,
                                        embbeding_dim, json_tokenizer_path)


def generate_model_operation(num_conv_layers,maxpool_strides,dropout, filter_sizes, num_filters, embbeding_dim, input_emb_matrix_path,
                             vocabulary_size_path, max_sequence_lenght_path, num_classes_path):
    return generate_model_component(num_conv_layers,maxpool_strides,dropout, filter_sizes, num_filters, embbeding_dim,
                                    input_emb_matrix_path, vocabulary_size_path, max_sequence_lenght_path,
                                    num_classes_path)


def move_data_pvc_operation(keras_model_path, x_train_path, x_val_path, y_train_path,
                            y_val_path, workdir):
    return move_data_pvc_component(keras_model_path, x_train_path, x_val_path, y_train_path,
                                   y_val_path, workdir)


def train_model_operation(keras_model_path, x_train_path, x_val_path, y_train_path, y_val_path, epochs, batch_size):
    return train_model_component(keras_model_path, x_train_path, x_val_path, y_train_path, y_val_path, epochs,
                                 batch_size)



def train_model_operation_dist(epochs, batch_size, namespace, workdir):
    now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    name = "tf-job-cnn-01" + now
    namespace = namespace
    workerNum = 1
    ttlSecondsAfterFinished = -1
    tfjobTimeoutMinutes = 60
    deleteAfterDone = 'False'
    activeDeadlineSeconds = -1
    backoffLimit = -1
    version = "v1"
    cleanPodPolicy = "Running"
    return train_model_component_dist(
        name,
        namespace,
        version,
        activeDeadlineSeconds,
        backoffLimit,
        cleanPodPolicy,
        ttlSecondsAfterFinished,
        deleteAfterDone,
        tfjobTimeoutMinutes,
        epochs,
        batch_size,
        workdir,
        workerNum,
        0,
        'False',
        'False'
    )






def deploy_model_operation_pvc(namespace, workdir):
    return deploy_model_component_pvc(namespace, workdir)


def deploy_model_operation_par(namespace, workdir):
    return deploy_model_component_par(namespace, workdir)


def main(params):
    print('Generating and executing CNN Text classifier pipeline ...')


    train_data_path = params.train_data_path
    test_data_path = params.test_data_path
    column_target_value = params.column_target_value
    column_text_value = params.column_text_value
    val_data_pct = params.val_data_pct
    gcp_bucket = params.gcp_bucket
    w2v_model_path = params.w2v_model_path
    embbeding_dim = params.embbeding_dim
    dropout = params.dropout
    filter_sizes = params.filter_sizes
    num_conv_layers = params.num_conv_layers
    maxpool_strides = params.maxpool_strides
    num_filters = params.num_filters
    batch_size = params.batch_size
    epochs = params.epochs
    num_words = params.num_words
    namespace = params.namespace
    workdir = params.workdir

    @dsl.pipeline(
        name='CNN Text classifier pipeline',
        description='Pipeline for training and deploying a CNN text classifier using w2v pretrained embeddings. - Heavyweight version'
    )
    def cnn_pipeline(
            dist_training=params.dist_training,
            gpu_support=params.gpu_support
    ):
        now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        workspace_name = 'cnn_txt_classifier' + now
        run_name = workspace_name + now

        prepare_data_task = prepare_data_operation(train_data_path,
                                                   test_data_path,
                                                   column_target_value,
                                                   column_text_value,
                                                   val_data_pct,
                                                   gcp_bucket)
        dataset_input_values_train = {
            "description": "Train data",
            "name": train_data_path,
            "owner": "velascoluis@google.com",
            "data_uri": 'gs://' + "train_data_path",
            "version": "v0.1",
            "query": "autogen",
            "labels": None
        }
        dataset_log_train_data_task = metadata_logger_operation(log_type='dataset', workspace_name=workspace_name,
                                                                run_name=run_name,
                                                                input_values=dataset_input_values_train)

        dataset_input_values_test = {
            "description": "Test data",
            "name": test_data_path,
            "owner": "velascoluis@google.com",
            "data_uri": 'gs://' + "test_data_path",
            "version": "v0.1",
            "query": "autogen",
            "labels": None
        }
        log_test_data_task = metadata_logger_operation(log_type='dataset', workspace_name=workspace_name,
                                                       run_name=run_name, input_values=dataset_input_values_test)

        prepare_emb_task = prepare_emb_operation(gcp_bucket,
                                                 num_words,
                                                 w2v_model_path,
                                                 embbeding_dim,
                                                 prepare_data_task.outputs['json_tokenizer_path'])

        generate_model_task = generate_model_operation(num_conv_layers,
                                                       maxpool_strides,
                                                       dropout,
                                                       filter_sizes,
                                                       num_filters,
                                                       embbeding_dim,
                                                       prepare_emb_task.outputs['output_emb_matrix_path'],
                                                       prepare_emb_task.outputs['vocabulary_size_path'],
                                                       prepare_data_task.outputs['max_sequence_lenght_path'],
                                                       prepare_data_task.outputs['num_classes_path'])

        model_input_values_cnn = {
            "description": "CNN Keras model",
            "name": "CNN 100d - 3 convolutions - 100 filters - softmax",
            "owner": "velascoluis@google.com",
            "model_uri": workdir,
            "version": "v0.1",
            "hyperparameters": "epochs:" + str(epochs) + " batch_size:" + str(batch_size) + " dropout:" + str(
                dropout) + "num filters:" + str(num_filters),
            "learning_rate": None,
            "layers": filter_sizes,
            "early_stop": True,
            "labels": None
        }

        log_test_data_task = metadata_logger_operation(log_type='model', workspace_name=workspace_name,
                                                       run_name=run_name, input_values=model_input_values_cnn).after(
            generate_model_task)

        with dsl.Condition(dist_training == 'yes'):
            move_data_pvc = move_data_pvc_operation(generate_model_task.outputs['output_keras_model_path'],
                                                    prepare_data_task.outputs['x_train_data_path'],
                                                    prepare_data_task.outputs['x_val_data_path'],
                                                    prepare_data_task.outputs['y_train_data_path'],
                                                    prepare_data_task.outputs['y_val_data_path'],
                                                    workdir).add_pvolumes(
                {workdir: dsl.PipelineVolume(pvc="kfpipeline-data-pvc")})
            train_time_start = datetime.datetime.utcnow()
            train_model_task_dist = train_model_operation_dist(epochs,
                                                               batch_size,
                                                               namespace, workdir).after(move_data_pvc)
            train_time_finish = datetime.datetime.utcnow()
            metrics_cnn_input_values = {
                "description": "Training metrics",
                "name": "training_metrics",
                "owner": "velascoluis@google.com",
                "metric_uri": workdir,
                "data_set_id": None,
                "model_id": None,
                "metrics_type": metadata.Metrics.TESTING,
                "values": {"train_start_time": train_time_start.strftime("%Y%m%d%H%M%S"),
                           "train_finish_time": train_time_finish.strftime("%Y%m%d%H%M%S")},
                "early_stop": True,
                "labels": None
            }
            log_cnn_metric_data_task = metadata_logger_operation(log_type='metrics', workspace_name=workspace_name,
                                                                 run_name=run_name,
                                                                 input_values=metrics_cnn_input_values).after(
                train_model_task_dist)
            deploy_model_task = deploy_model_operation_pvc(namespace, workdir).after(train_model_task_dist)

        with dsl.Condition(dist_training == 'no'):
            train_time_start = datetime.datetime.utcnow()
            with dsl.Condition(gpu_support == 'no'):
                train_model_task = train_model_operation(generate_model_task.outputs['output_keras_model_path'],
                                                         prepare_data_task.outputs['x_train_data_path'],
                                                         prepare_data_task.outputs['x_val_data_path'],
                                                         prepare_data_task.outputs['y_train_data_path'],
                                                         prepare_data_task.outputs['y_val_data_path'],
                                                         batch_size, epochs)
                train_time_finish = datetime.datetime.utcnow()
                metrics_cnn_input_values = {
                    "description": "Training metrics",
                    "name": "training_metrics",
                    "owner": "velascoluis@google.com",
                    "metric_uri": workdir,
                    "data_set_id": None,
                    "model_id": None,
                    "metrics_type": metadata.Metrics.TESTING,
                    "values": {"train_start_time": train_time_start.strftime("%Y%m%d%H%M%S"),
                               "train_finish_time": train_time_finish.strftime("%Y%m%d%H%M%S")},
                    "early_stop": True,
                    "labels": None
                }
                log_cnn_metric_data_task = metadata_logger_operation(log_type='metrics', workspace_name=workspace_name,
                                                                     run_name=run_name,
                                                                     input_values=metrics_cnn_input_values).after(train_model_task)
                deploy_model_task = deploy_model_operation_par(namespace,
                                                               train_model_task.outputs['output_trained_model_path'])
            with dsl.Condition(gpu_support == 'yes'):
                train_model_task = train_model_operation(generate_model_task.outputs['output_keras_model_path'],
                                                             prepare_data_task.outputs['x_train_data_path'],
                                                             prepare_data_task.outputs['x_val_data_path'],
                                                             prepare_data_task.outputs['y_train_data_path'],
                                                             prepare_data_task.outputs['y_val_data_path'],
                                                             batch_size, epochs).set_gpu_limit(1)
                train_time_finish = datetime.datetime.utcnow()
                metrics_cnn_input_values = {
                    "description": "Training metrics",
                    "name": "training_metrics",
                    "owner": "velascoluis@google.com",
                    "metric_uri": workdir,
                    "data_set_id": None,
                    "model_id": None,
                    "metrics_type": metadata.Metrics.TESTING,
                    "values": {"train_start_time": train_time_start.strftime("%Y%m%d%H%M%S"),
                               "train_finish_time": train_time_finish.strftime("%Y%m%d%H%M%S")},
                    "early_stop": True,
                    "labels": None
                }
                log_cnn_metric_data_task = metadata_logger_operation(log_type='metrics', workspace_name=workspace_name,
                                                                     run_name=run_name,
                                                                     input_values=metrics_cnn_input_values).after(
                    train_model_task)
                deploy_model_task = deploy_model_operation_par(namespace,
                                                               train_model_task.outputs['output_trained_model_path'])


    # Generate .zip file
    pipeline_func = cnn_pipeline
    pipeline_filename = pipeline_func.__name__ + '.kf_pipeline_containers.zip'
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
    experiment = client.create_experiment('dev-testing-containers-tfjob')
    experiment_name = 'dev-testing-containers-tfjob'
    arguments = {
        "dist_training": params.dist_training,
        "gpu_support": params.gpu_support
    }

    run_name = pipeline_func.__name__ + ' run'
    run_result = client.run_pipeline(
        experiment_id=experiment.id,
        job_name=run_name,
        pipeline_package_path=pipeline_filename,
        params=arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Component-based build-train-deploy pipeline')
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--column_target_value', type=str)
    parser.add_argument('--column_text_value', type=str)
    parser.add_argument('--val_data_pct', type=float)
    parser.add_argument('--num_words', type=int)
    parser.add_argument('--gcp_bucket', type=str)
    parser.add_argument('--json_tokenizer_path', type=str)
    parser.add_argument('--w2v_model_path', type=str)
    parser.add_argument('--embbeding_dim', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--filter_sizes', type=str)
    parser.add_argument('--num_filters', type=int)
    parser.add_argument('--num_conv_layers', type=int)
    parser.add_argument('--maxpool_strides', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--host', type=str)
    parser.add_argument('--client_id', type=str)
    parser.add_argument('--other_client_id', type=str)
    parser.add_argument('--other_client_secret', type=str)
    parser.add_argument('--namespace', type=str)
    parser.add_argument('--workdir', type=str)
    parser.add_argument('--dist_training', type=str)
    parser.add_argument('--gpu_support', type=str)
    params = parser.parse_args()
    main(params)
