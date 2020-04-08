# CNN Text Classifier - Deploy model

This component performs deploys a trained model using KFServing
- It uses the KFServingClient SDK

### Input data
Interface of the component:
```
- {name: namespace,               type: String,     default: 'None',    description: kubeflow namespace}
- {name: trained_model_path,      type: String,     default: 'None',    description: Load trained model data path}
```
