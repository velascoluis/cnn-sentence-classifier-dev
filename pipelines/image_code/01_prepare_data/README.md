# CNN Text Classifier - Data preparation

This component performs data preparation tasks on input data. It does:
- Load data from GCS bucket
- Drop rows with NULLs values
- Deduplicate rows
- Generate validation data applying sampling on the input data
- Tokenize the data and generate text sequences

### Input data
Interface of the component:
```
inputs:
- {name: Train data path,       type: String,   default: 'None',                        description: path of the csv file with the trainning data}
- {name: Test data path,        type: String,   default: 'None',                        description: path of the csv file with the test data}
- {name: Column target value,   type: String,   default: 'None',                        description: name of the column name (label) we want to predict - top csv column}
- {name: Column text value,     type: String,   default: 'None',                        description: name of the input text column name we want to analyze - top csv column}
- {name: Val data pct,          type: Float,    default: '0.2',                         description: train test data split percentage}
- {name: Max num words,         type: Integer,  default: '20000',                       description: the maximum number of words to keep based on word frequency}
- {name: GCP Bucket name,       type: String,   default: 'None',                        description: Name of the GCP bucket with input data}

outputs:
- {name: json_tokenizer_path,        type: String,   description: Tokenizer serialized in JSON format}
- {name: x_train_data_path,          type: String,   description: path of the trainning data - text}
- {name: x_val_data_path,            type: String,   description: path of validation data - text}
- {name: y_train_data_path,          type: String,   description: path of the trainning data - labels }
- {name: y_val_data_path,            type: String,   description: ath of validation data - labels}
- {name: max_sequence_lenght_path,   type: String,   description: maximun value of an observation}
- {name: num_classes_path,           type: String,   description: calculated num of classes to predict}
- {name: classifier_values,          type: String,   description: Serialized values to classify}
```
### Example of input data

Train/test data format:
```
LABEL_TEXT,                             LABEL_PREDICT
Lorem ipsum dolor sit amet,             Class1
consectetur adipiscing elit,            Class2
sed do eiusmod tempor incididunt,       ClassN
...
```