import argparse
from datetime import datetime
import logging
import json
from kubeflow.metadata import metadata
import retrying

DATASET = 'dataset'
MODEL = 'model'
METRIC = 'metrics'
METADATA_STORE_HOST = "metadata-grpc-service.kubeflow"
METADATA_STORE_PORT = 8080



def get_or_create_workspace(ws_name):
    return metadata.Workspace(
        store=metadata.Store(grpc_host=METADATA_STORE_HOST, grpc_port=METADATA_STORE_PORT),
        name=ws_name,
        description="CNN Text classifier metadata workspace",
        labels={"n1": "v1"})



def get_or_create_workspace_run(md_workspace, run_name):
    return metadata.Run(
        workspace=md_workspace,
        name=run_name,
        description="Metadata run for workflow %s" % run_name,
    )


#This decorator is to avoid race conditions when executing this component in paralell
@retrying.retry(stop_max_delay=200000)
def log_model_info(ws, ws_run, description, name, owner, model_uri, version, hyperparameters, learning_rate, layers,
                   early_stop, labels):
    exec = metadata.Execution(
        name="Execution" + datetime.utcnow().isoformat("T"),
        workspace=ws,
        run=ws_run,
        description="Model log exec.",
    )
    model_log = exec.log_input(
        metadata.Model(
            description=description,
            name=name,
            owner=owner,
            uri=model_uri,
            version=version,
            hyperparameters=hyperparameters,
            learning_rate=learning_rate,
            layers=layers,
            early_stop=early_stop,
            labels=labels,
        ))

@retrying.retry(stop_max_delay=200000)
def log_metric_info(ws, ws_run, description, name, owner, metric_uri, data_set_id, model_id, metrics_type, values,
                    labels):
    exec = metadata.Execution(
        name="Execution" + datetime.utcnow().isoformat("T"),
        workspace=ws,
        run=ws_run,
        description="Metric log exec.",
    )
    metric_log = exec.log_input(
        metadata.Metrics(
            description=description,
            name=name,
            owner=owner,
            uri=metric_uri,
            data_set_id=data_set_id,
            model_id=model_id,
            metrics_type=metrics_type,
            values=values,
            labels=labels
        ))

@retrying.retry(stop_max_delay=180000)
def log_dataset_info(ws, ws_run, description, name, owner, data_uri, version, query, labels):
    exec = metadata.Execution(
        name="Execution" + datetime.utcnow().isoformat("T"),
        workspace=ws,
        run=ws_run,
        description="Dataset log exec.",
    )
    dataset_log = exec.log_input(
        metadata.DataSet(
            description=description,
            name=name,
            owner=owner,
            uri=data_uri,
            version=version,
            query=query,
            labels=labels
        ))


def main(params):
    ws = get_or_create_workspace(params.workspace_name)
    ws_run = get_or_create_workspace_run(ws, params.run_name)

    if params.log_type.lower() == DATASET:
        description = params.input_values["description"]
        name = params.input_values["name"]
        owner = params.input_values["owner"]
        data_uri = params.input_values["data_uri"]
        version = params.input_values["version"]
        query = params.input_values["query"]
        labels = params.input_values["labels"]
        log_dataset_info(ws, ws_run, description, name, owner, data_uri, version, query, labels)

    elif params.log_type.lower() == MODEL:
        description = params.input_values["description"]
        name = params.input_values["name"]
        owner = params.input_values["owner"]
        model_uri = params.input_values["model_uri"]
        version = params.input_values["version"]
        hyperparameters = params.input_values["hyperparameters"]
        learning_rate = params.input_values["learning_rate"]
        layers = params.input_values["layers"]
        early_stop = params.input_values["early_stop"]
        labels = params.input_values["labels"]
        log_model_info(ws, ws_run, description, name, owner, model_uri, version, hyperparameters, learning_rate, layers,
                       early_stop, labels)

    elif params.log_type.lower() == METRIC:
        description = params.input_values["description"]
        name = params.input_values["name"]
        owner = params.input_values["owner"]
        metric_uri = params.input_values["metric_uri"]
        data_set_id = params.input_values["data_set_id"]
        model_id = params.input_values["model_id"]
        metrics_type = params.input_values["metrics_type"]
        values = params.input_values["values"]
        labels = params.input_values["labels"]
        log_metric_info(ws, ws_run, description, name, owner, metric_uri, data_set_id, model_id, metrics_type, values,
                    labels)
    else:
        logging.warning("Error: Unknown metadata logging type %s", params.log_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='01. Gather dataset step')
    parser.add_argument('--log_type', type=str, default='None')
    parser.add_argument('--workspace-name', type=str, default='None')
    parser.add_argument('--run-name', type=str, default='None')
    parser.add_argument('--input_values', type=json.loads, default='None')
    params = parser.parse_args()
    main(params)
