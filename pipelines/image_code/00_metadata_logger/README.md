# CNN Text Classifier - Metadata logger

This component exposes an interface for logging metadata into the kubeflow ML Metadata (MLMD):
- It retrieves or generate a new workspace
- It retrieves or generate a new run
- Insert the desired metadata (dataset, metric or model)


### Input data
Interface of the component:
```
inputs:
- {name: log_type,                type: String,   default: 'None',        description: Values are dataset model or metric}
- {name: workspace-name,          type: String,   default: 'None',        description: Name of the workspace}
- {name: run-name,                type: String,   default: 'None',        description: Name of the run}
- {name: input_values,            type: dict,     default: 'None',        description: Dict with input values}

```
### Example of call:

Log training time metrics:
```
metrics_cnn_input_values = {
                    "description": "Training metrics",
                    "name": "training_metrics",
                    "owner": "velascoluis@google.com",
                    "metric_uri": workdir,
                    "data_set_id": dataset_id,
                    "model_id": model_id,
                    "metrics_type": metadata.Metrics.TESTING,
                    "values": {
                                    "train_start_time": train_time_start.strftime("%Y%m%d%H%M%S"),
                                    "train_finish_time": train_time_finish.strftime("%Y%m%d%H%M%S")},
                    "early_stop": True,
                    "labels": None
                }

metadata_logger_operation(log_type='metrics', workspace_name=workspace_name,run_name=run_name,
                            input_values=metrics_cnn_input_values)
```