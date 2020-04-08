# CNN Text Classifier - Train model

This component performs training of model:
- It compiles the model using:
    - Loss function = categorical crossentropy
    - Optimizer = adam

### Input data
Interface of the component:
```
- {name: keras_model_path,        type: String,     default: 'None',    description: Load keras model path}
- {name: x_train_path,            type: String,     default: 'None',    description: Load x train data path}
- {name: x_val_path,              type: String,     default: 'None',    description: Load x val data path}
- {name: y_train_path,            type: String,    default: 'None',    description: Load y train data path}
- {name: y_val_path,              type: String,     default: 'None',    description: Load y train data path}
- {name: epochs,                  type: Integer,     default: 'None',    description: Epochs}
- {name: batch_size,              type: Integer,     default: 'None',    description: Batch}
```
