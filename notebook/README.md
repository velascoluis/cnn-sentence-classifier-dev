# EdA and hyperparameter tuning notebook

This notebooks tries to showcase a cannonical example in building a ML model.
The problem we will try to solve is building a sentence classification model, we will use movie data from wikipedia and try to predict the genre.
In general, the notebook shows how:
- Run data preparation (dedup, balancing ..)
- Build a keras model
- Train locally the keras model
- Wrap the code and package it in image using kaniko (via fairing)
- Generate a experiment definition file for hyperparameter tuning
- Launch a remote katib job to find the optimal parameters for the notebook