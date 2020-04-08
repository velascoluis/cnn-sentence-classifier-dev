# EdA and hyperparameter tuning notebook

This notebooks tries to showcase an example in building a ML model.
The problem we will try to solve is building a sentence classification model, we will use movie data from wikipedia and try to predict the genre.
In general, the notebook shows how:
- Run data preparation (dedup, balancing ..)
- Build a keras model
- Train locally the keras model
- Wrap the code and package it in image using kaniko (via fairing)
- Generate a experiment definition file for hyperparameter tuning
- Launch a remote katib job to find the optimal parameters for the notebook
Altought the performance is not stellar it gives a nice oveerview of how to use kubeflow to perform initial steps on model generation
---
**Notebook setup:**
- Deploy a kubeflow notebook following [this](https://www.kubeflow.org/docs/notebooks/setup/)
- The input data used for this notebook can be found [here](https://www.kaggle.com/jrobischon/wikipedia-movie-plots), kudos to [justinR](https://www.kaggle.com/jrobischon)  for preparing this dataset.
- Upload data to GCS, make sure you KSA/SA bindings are in place and have read access [permisions](https://www.kubeflow.org/docs/gke/authentication/) to the bucket 
    - example setup:
        ```
        train_data_path = 'data/wiki_movie_plots_deduped.csv'
        test_data_path = 'data/wiki_movie_plots_deduped_test.csv'
        gcp_bucket = 'velascoluis-test'
      ```
- Generate or download a pre-trained word2vec embedding model. The notebook uses [GloVe](https://nlp.stanford.edu/projects/glove/).
    note: for transforming GloVe to word2vec:
    ```
    from gensim.test.utils import datapath, get_tmpfile
    from gensim.models import KeyedVectors
    from gensim.scripts.glove2word2vec import glove2word2vec
    glove_file = datapath('glove.6B.100d.txt')
    tmp_file = get_tmpfile('word2vec100d.txt')
    _ = glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format
```      