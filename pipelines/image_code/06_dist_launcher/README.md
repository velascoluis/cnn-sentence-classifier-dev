# CNN Text Classifier - Distributed training launcher

This component generates a TFJOb spec (worker, master and PS) and then launches a container with the distributed code.
- Note: Code is an adaptation of https://github.com/kubeflow/pipelines/blob/master/components/kubeflow/launcher/src/launch_tfjob.py 

### Input data
Interface of the component:
```
  - {name: name, type: String}
  - {name: namespace,type: String}
  - {name: version,type: String}
  - {name: activeDeadlineSeconds,type: Integer}
  - {name: backoffLimit,type: Integer}
  - {name: cleanPodPolicy,type: String}
  - {name: ttlSecondsAfterFinished,type: Integer}
  - {name: deleteAfterDone,type: bool}
  - {name: tfjobTimeoutMinutes,type: Integer}
  - {name: epochs,type: Integer}
  - {name: batch_size,type: Integer}
  - {name: workdir,type: String}
  - {name: worker_num_replicas,type: Integer}
  - {name: ps_num_replicas,type: Integer}
  - {name: master,type: bool}
  - {name: evaluator,type: bool}
```
