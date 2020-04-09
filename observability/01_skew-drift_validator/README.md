# KF Pipelines - Observability framework - Data drift validator

This component exposes TFDV skew and drift validation as a component a reusable component

### Input data
Interface of the component:
```
- {name: mode,                      type: String,   default: 'None',        description: skew or drift}
- {name: gcp_bucket,                type: String,   default: 'None',        description: GCP bucket with data}
- {name: control_set_path,          type: String,   default: 'None',        description: Path with control data in GCP}
- {name: treatment_set_path,        type: String,   default: 'None',        description: Path with treatment data in GCP}
- {name: feature_list,              type: String,   default: 'None',        description: Features to evaluate}
- {name: Linf_value,                type: float,    default: 'None',         description: L-inf norm value for evaluation}

```
### Example of call:
```
skew_drift_validator.py 
--mode skew 
--gcp_bucket velascoluis-test 
--control_set_path data/test-data.csv 
--treatment_set_path data/test-data.csv 
--feature_list '["Feature0","Feature3"]' 
--Linf_value 0.01
```