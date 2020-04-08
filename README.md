# cnn-text-classifier-dev
This repository contains examples of data science CUJs resolved using kubeflow.
In particular, as of now, this repo features:
- Notebooks:
    - Exploratory data analysis
    - Local training with keras
    - Launching a katib hyperparameter tuning job using fairing + katib
- Pipelines:
    - Code refactor and generation of lightweight pipeline
    - Code refactor into components
        - Using CRD operators (TFJob) - Launcher
        - Using hardware accelerators
    - Metadata component to recalibrate models based on data drift/skew using TFDV + MLMD        