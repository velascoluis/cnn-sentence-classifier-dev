# CNN Text Classifier - Prepare embeddings

This component load a pre-trained gensim word2vec model (stored in GCS)
Then it takes a tokenized text and generates and returns a dense matrix where each row is the w2v vector associated with the token
- There are several pre-trained word2vec embeddings online, for instance:
    - https://code.google.com/archive/p/word2vec/
    - https://nlp.stanford.edu/projects/glove/
### Input data
Interface of the component:
```
inputs:
- {name: gcp_bucket,            type: String,     default: 'None',                            description: Name of the GCP bucket with input data}
- {name: num_words,             type: Integer,    default: '20000',                           description: the maximum number of words to keep based on word frequency}
- {name: w2v_model_path,        type: String,     default: 'None',                            description: pre-generated w2v gensim model}
- {name: embbeding_dim,         type: Integer,    default: '100',                             description: dimension of the dense vectors}
- {name: json_tokenizer_path,   type: String,     default: 'None',                            description: Tokenizer object to load}



outputs:
- {name: output_emb_matrix_path,        type: String,   description: Emb matrix}
- {name: vocabulary_size_path,          type: String,   description: Vocabulary size}
```
