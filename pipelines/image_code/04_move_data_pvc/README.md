# CNN Text Classifier - Move data to Volume

This component moves data from pipeline output parametets to a PVC, we need to do that in order to ensure that a distributed TFJOB would have access to the data+artifacts
- Note: This component is only executed if performing distributed training
### Input data
Interface of the component:
```
inputs:
- {name: keras_model_path,        type: String,     default: 'None',    description: Load keras model path}
- {name: x_train_path,            type: String,     default: 'None',    description: Load x train data path}
- {name: x_val_path,              type: String,     default: 'None',    description: Load x val data path}
- {name: y_train_path,            type: String,     default: 'None',     description: Load y train data path}
- {name: y_val_path,              type: String,     default: 'None',    description: Load y train data path}
- {name: workdir,                 type: String,     default: 'None',    description: PVC workdir}
```
