# CNN Text Classifier - Generate model

This component generates the metadata for the deep neuronal network model. It does:
- Generate a CNN Keras (functional API) with the following architecture:
    - Embedding layer + Convolution layer + maxpool layer + Dense (outputs)
    - Model arch. is based on https://arxiv.org/pdf/1408.5882.pdf
    - Notes:
        - kernel sizes are automatically generated following the series: [3,5,7,9 ...]
        - Ouput layer is calculated based on the num classes to predict
    
### Input data
Interface of the component:
```
inputs:
- {name: num_conv_layers,          type: Integer,    default: 'None',    description: Num convolutions}
- {name: maxpool_strides,          type: Integer,    default: 'None',    description: Max pool strides}
- {name: dropout,                  type: Float,      default: 'None',    description: DNN droput regularizer value}
- {name: filter_sizes,             type: String,     default: 'None',    description: Filter sizes}
- {name: embbeding_dim,            type: Integer,    default: 'None',    description: dimension of the dense vectors}
- {name: input_emb_matrix_path,    type: String,     default: 'None',    description: Load emb matrix path}
- {name: vocabulary_size_path,     type: String,     default: 'None',    description: Load vocabulary size path}
- {name: max_sequence_lenght_path, type: String,     default: 'None',    description: Load sequence length path}
- {name: num_classes_path,         type: String,     default: 'None',    description: Load num classes path}


outputs:
- {name: output_keras_model_path,        type: String,   description: Output keras model}
```
### Example of model ouput architecture

Model output: (example with 5 convolutions)
```
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 105)]        0                                            
__________________________________________________________________________________________________
embedding (Embedding)           (None, 105, 100)     918200      input_1[0][0]                    
__________________________________________________________________________________________________
reshape (Reshape)               (None, 105, 100, 1)  0           embedding[0][0]                  
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 103, 1, 100)  30100       reshape[0][0]                    
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 102, 1, 100)  40100       reshape[0][0]                    
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 101, 1, 100)  50100       reshape[0][0]                    
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 99, 1, 100)   70100       reshape[0][0]                    
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 97, 1, 100)   90100       reshape[0][0]                    
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 1, 1, 100)    0           conv2d[0][0]                     
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 1, 1, 100)    0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)  (None, 1, 1, 100)    0           conv2d_2[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)  (None, 1, 1, 100)    0           conv2d_3[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)  (None, 1, 1, 100)    0           conv2d_4[0][0]                   
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 5, 1, 100)    0           max_pooling2d[0][0]              
                                                                 max_pooling2d_1[0][0]            
                                                                 max_pooling2d_2[0][0]            
                                                                 max_pooling2d_3[0][0]            
                                                                 max_pooling2d_4[0][0]            
__________________________________________________________________________________________________
flatten (Flatten)               (None, 500)          0           concatenate[0][0]                
__________________________________________________________________________________________________
dropout (Dropout)               (None, 500)          0           flatten[0][0]                    
__________________________________________________________________________________________________
dense (Dense)                   (None, 10)           5010        dropout[0][0]                    
```