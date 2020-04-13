# Observability/Production utils KF components

Repository of various observability components for ML production pipelines, mostly based on TFX and TFDV.  
Current ideas are around:

- :heavy_check_mark: Data skew(train)/drift(predict) checker
    - ``` 01_skew-drift-validator```
- :white_check_mark: General data preparation components
- :white_check_mark: Uber component for relaunching models based on different inputs (accuracy metrics, schema changes, drift ..)
- :white_check_mark: Ground truth checker
- :white_check_mark: Continuous model improvement

  