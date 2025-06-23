# -*- coding: utf-8 -*-
"""
@authors: Mathias and Carlos
"""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras.regularizers import l2
import numpy as np

#These networks are based on NetTCR 2.1 by Alessandro Montemurro

def CNN_1D_global_max(dropout_rate, seed, hidden_sizes=[64,32],
                                            nfilters=16,
                                            input_dict={'pep':{'ks':[1,3,5,7,9],'dim':12, 'embed_dim': 20},
                                                        'a1':{'ks':[1,3,5,7,9],'dim':7, 'embed_dim': 20},
                                                        'a2':{'ks':[1,3,5,7,9],'dim':8, 'embed_dim': 20},
                                                        'a3':{'ks':[1,3,5,7,9],'dim':22, 'embed_dim': 20},
                                                        'b1':{'ks':[1,3,5,7,9],'dim':6, 'embed_dim': 20},
                                                        'b2':{'ks':[1,3,5,7,9],'dim':7, 'embed_dim': 20},
                                                        'b3':{'ks':[1,3,5,7,9],'dim':23, 'embed_dim': 20}}):

    """A generic 1D CNN model initializer function, used for basic NetTCR-like architectures"""

    assert len(input_dict) > 0, "input_names cannot be empty"
    assert len(hidden_sizes) > 0, "Must provide at least one hidden layer size"


    conv_activation = "relu"
    dense_activation = "sigmoid"

    input_names = [name for name in input_dict]

    # Inputs
    inputs = [keras.Input(shape = (input_dict[name]['dim'], input_dict[name]['embed_dim']), name=name) for name in input_names]

    cnn_layers = []
    # Define CNN layers inputs
    for name, inp in zip(input_names, inputs):
        for k in input_dict[name]['ks']:  # kernel sizes
            conv = layers.Conv1D(filters=nfilters, kernel_size=k, padding="same", name=f"{name}_conv_{k}")(inp)
            bn = layers.BatchNormalization()(conv)
            activated = layers.Activation(conv_activation)(bn)  # Apply activation after BatchNorm
            cnn_layers.append(activated)

    pool_layers = [layers.GlobalMaxPooling1D()(p) for p in cnn_layers]

    # Concatenate all max pooling layers to a single layer
    cat = layers.Concatenate()(pool_layers)

    # Dropout - Required to prevent overfitting
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)

    # Dense layer
    dense = [layers.Dense(hidden_sizes[0], activation = dense_activation)(cat_dropout)]

    if len(hidden_sizes) > 1:
        for i in range(1, len(hidden_sizes)):
            dense.append(layers.Dense(hidden_sizes[i], activation = dense_activation)(dense[i-1]))

    # Output layer
    out = layers.Dense(1,activation = "sigmoid")(dense[-1])

    # Prepare model object
    model = keras.Model(inputs = inputs, outputs = out)

    return model

def CNN_1D_global_max_embed(dropout_rate, seed, 
                                            hla_embedding_dim=66, 
                                            hidden_sizes=[64,32],
                                            nfilters=16,
                                            input_dict={'pep':{'ks':[1,3,5,7,9],'dim':12, 'embed_dim': 20},
                                                        'a1':{'ks':[1,3,5,7,9],'dim':7, 'embed_dim': 20},
                                                        'a2':{'ks':[1,3,5,7,9],'dim':8, 'embed_dim': 20},
                                                        'a3':{'ks':[1,3,5,7,9],'dim':22, 'embed_dim': 20},
                                                        'b1':{'ks':[1,3,5,7,9],'dim':6, 'embed_dim': 20},
                                                        'b2':{'ks':[1,3,5,7,9],'dim':7, 'embed_dim': 20},
                                                        'b3':{'ks':[1,3,5,7,9],'dim':23, 'embed_dim': 20}}):

    """A generic 1D CNN model initializer function, used for basic NetTCR-like architectures"""

    assert len(input_dict) > 0, "input_names cannot be empty"
    assert len(hidden_sizes) > 0, "Must provide at least one hidden layer size"

    conv_activation = "relu"
    dense_activation = "sigmoid"

    input_names = [name for name in input_dict]

    # Inputs
    inputs = [keras.Input(shape = (input_dict[name]['dim'], input_dict[name]['embed_dim']), name=name) for name in input_names]

    cnn_layers = []
    # Define CNN layers inputs
    for name, inp in zip(input_names, inputs):
        for k in input_dict[name]['ks']:  # kernel sizes
            conv = layers.Conv1D(filters=nfilters, kernel_size=k, padding="same", name=f"{name}_conv_{k}")(inp)
            bn = layers.BatchNormalization()(conv)
            activated = layers.Activation(conv_activation)(bn)  # Apply activation after BatchNorm
            cnn_layers.append(activated)

    pool_layers = [layers.GlobalMaxPooling1D()(p) for p in cnn_layers]

    # Concatenate all max pooling layers to a single layer
    cat = layers.Concatenate()(pool_layers)

    # Dropout - Required to prevent overfitting
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)

    embed_inp = keras.Input(shape = (hla_embedding_dim,), name ="embedding")
    final_cat = layers.Concatenate(name = "inp_cat")([cat_dropout, embed_inp])

    # Dense layer
    dense = [layers.Dense(hidden_sizes[0], activation = dense_activation)(final_cat)]

    if len(hidden_sizes) > 1:
        for i in range(1, len(hidden_sizes)):
            dense.append(layers.Dense(hidden_sizes[i], activation = dense_activation)(dense[i-1]))

    # Output layer
    out = layers.Dense(1,activation = "sigmoid")(dense[-1])

    # Prepare model object
    model = keras.Model(inputs = inputs + [embed_inp], outputs = out)

    return model


# Main baseline architecture for all paired-chain CDR sequences
def CNN_CDR123_1D_baseline(dropout_rate, seed, conv_activation = "relu", dense_activation = "sigmoid",
                           embed_dim = 20, nr_of_filters_1 = 16, max_lengths = None):

    # Max.length of the sequences from the dataset
    if max_lengths:
        a1_max = max_lengths[0]
        a2_max = max_lengths[1]
        a3_max = max_lengths[2]
        b1_max = max_lengths[3]
        b2_max = max_lengths[4]
        b3_max = max_lengths[5]
        pep_max = max_lengths[6]
    else:
        a1_max = 7
        a2_max = 8
        a3_max = 22
        b1_max = 6
        b2_max = 7
        b3_max = 23
        pep_max = 12
    
    # Input dimensions
    pep = keras.Input(shape = (pep_max, embed_dim), name ="pep")
    a1 = keras.Input(shape = (a1_max, embed_dim), name ="a1")
    a2 = keras.Input(shape = (a2_max, embed_dim), name ="a2")
    a3 = keras.Input(shape = (a3_max, embed_dim), name ="a3")
    b1 = keras.Input(shape = (b1_max, embed_dim), name ="b1")
    b2 = keras.Input(shape = (b2_max, embed_dim), name ="b2")
    b3 = keras.Input(shape = (b3_max, embed_dim), name ="b3")
    
    # Trained in a pan-specific setup (padding="same" for keeping original input sizes)

    # CNN layers for each feature and the different 5 kernel-sizes
    pep_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_pep_1_conv")(pep)
    pep_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_pep_3_conv")(pep)
    pep_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_pep_5_conv")(pep)
    pep_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_pep_7_conv")(pep)
    pep_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_pep_9_conv")(pep)
    
    a1_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a1_1_conv")(a1)
    a1_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a1_3_conv")(a1)
    a1_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a1_5_conv")(a1)
    a1_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a1_7_conv")(a1)
    a1_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a1_9_conv")(a1)
    
    a2_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a2_1_conv")(a2)
    a2_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a2_3_conv")(a2)
    a2_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a2_5_conv")(a2)
    a2_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a2_7_conv")(a2)
    a2_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a2_9_conv")(a2)
    
    a3_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a3_1_conv")(a3)
    a3_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a3_3_conv")(a3)
    a3_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a3_5_conv")(a3)
    a3_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a3_7_conv")(a3)
    a3_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a3_9_conv")(a3)
    
    b1_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b1_1_conv")(b1)
    b1_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b1_3_conv")(b1)
    b1_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b1_5_conv")(b1)
    b1_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b1_7_conv")(b1)
    b1_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b1_9_conv")(b1)
    
    b2_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b2_1_conv")(b2)
    b2_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b2_3_conv")(b2)
    b2_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b2_5_conv")(b2)
    b2_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b2_7_conv")(b2)
    b2_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b2_9_conv")(b2)
    
    b3_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b3_1_conv")(b3)
    b3_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b3_3_conv")(b3)
    b3_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b3_5_conv")(b3)
    b3_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b3_7_conv")(b3)
    b3_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b3_9_conv")(b3) 
    
    # GlobalMaxPooling: takes the maximum value along the entire sequence for each feature map (16 filters)
    pep_1_pool = layers.GlobalMaxPooling1D(name = "first_pep_1_pool")(pep_1_CNN)
    pep_3_pool = layers.GlobalMaxPooling1D(name = "first_pep_3_pool")(pep_3_CNN)
    pep_5_pool = layers.GlobalMaxPooling1D(name = "first_pep_5_pool")(pep_5_CNN)
    pep_7_pool = layers.GlobalMaxPooling1D(name = "first_pep_7_pool")(pep_7_CNN)
    pep_9_pool = layers.GlobalMaxPooling1D(name = "first_pep_9_pool")(pep_9_CNN)
    
    a1_1_pool = layers.GlobalMaxPooling1D(name = "first_a1_1_pool")(a1_1_CNN)
    a1_3_pool = layers.GlobalMaxPooling1D(name = "first_a1_3_pool")(a1_3_CNN)
    a1_5_pool = layers.GlobalMaxPooling1D(name = "first_a1_5_pool")(a1_5_CNN)
    a1_7_pool = layers.GlobalMaxPooling1D(name = "first_a1_7_pool")(a1_7_CNN)
    a1_9_pool = layers.GlobalMaxPooling1D(name = "first_a1_9_pool")(a1_9_CNN)
    
    a2_1_pool = layers.GlobalMaxPooling1D(name = "first_a2_1_pool")(a2_1_CNN)
    a2_3_pool = layers.GlobalMaxPooling1D(name = "first_a2_3_pool")(a2_3_CNN)
    a2_5_pool = layers.GlobalMaxPooling1D(name = "first_a2_5_pool")(a2_5_CNN)
    a2_7_pool = layers.GlobalMaxPooling1D(name = "first_a2_7_pool")(a2_7_CNN)
    a2_9_pool = layers.GlobalMaxPooling1D(name = "first_a2_9_pool")(a2_9_CNN)
    
    a3_1_pool = layers.GlobalMaxPooling1D(name = "first_a3_1_pool")(a3_1_CNN)
    a3_3_pool = layers.GlobalMaxPooling1D(name = "first_a3_3_pool")(a3_3_CNN)
    a3_5_pool = layers.GlobalMaxPooling1D(name = "first_a3_5_pool")(a3_5_CNN)
    a3_7_pool = layers.GlobalMaxPooling1D(name = "first_a3_7_pool")(a3_7_CNN)
    a3_9_pool = layers.GlobalMaxPooling1D(name = "first_a3_9_pool")(a3_9_CNN)
    
    b1_1_pool = layers.GlobalMaxPooling1D(name = "first_b1_1_pool")(b1_1_CNN)
    b1_3_pool = layers.GlobalMaxPooling1D(name = "first_b1_3_pool")(b1_3_CNN)
    b1_5_pool = layers.GlobalMaxPooling1D(name = "first_b1_5_pool")(b1_5_CNN)
    b1_7_pool = layers.GlobalMaxPooling1D(name = "first_b1_7_pool")(b1_7_CNN)
    b1_9_pool = layers.GlobalMaxPooling1D(name = "first_b1_9_pool")(b1_9_CNN)
    
    b2_1_pool = layers.GlobalMaxPooling1D(name = "first_b2_1_pool")(b2_1_CNN)
    b2_3_pool = layers.GlobalMaxPooling1D(name = "first_b2_3_pool")(b2_3_CNN)
    b2_5_pool = layers.GlobalMaxPooling1D(name = "first_b2_5_pool")(b2_5_CNN)
    b2_7_pool = layers.GlobalMaxPooling1D(name = "first_b2_7_pool")(b2_7_CNN)
    b2_9_pool = layers.GlobalMaxPooling1D(name = "first_b2_9_pool")(b2_9_CNN)
    
    b3_1_pool = layers.GlobalMaxPooling1D(name = "first_b3_1_pool")(b3_1_CNN)
    b3_3_pool = layers.GlobalMaxPooling1D(name = "first_b3_3_pool")(b3_3_CNN)
    b3_5_pool = layers.GlobalMaxPooling1D(name = "first_b3_5_pool")(b3_5_CNN)
    b3_7_pool = layers.GlobalMaxPooling1D(name = "first_b3_7_pool")(b3_7_CNN)
    b3_9_pool = layers.GlobalMaxPooling1D(name = "first_b3_9_pool")(b3_9_CNN)

    # Concatenation of all MaxPool outputs from all features and kernel-sizes
    cat = layers.Concatenate(name = "first_cat")([pep_1_pool, pep_3_pool, pep_5_pool, pep_7_pool, pep_9_pool,
                                a1_1_pool, a1_3_pool, a1_5_pool, a1_7_pool, a1_9_pool,
                                a2_1_pool, a2_3_pool, a2_5_pool, a2_7_pool, a2_9_pool,
                                a3_1_pool, a3_3_pool, a3_5_pool, a3_7_pool, a3_9_pool,
                                b1_1_pool, b1_3_pool, b1_5_pool, b1_7_pool, b1_9_pool,
                                b2_1_pool, b2_3_pool, b2_5_pool, b2_7_pool, b2_9_pool,
                                b3_1_pool, b3_3_pool, b3_5_pool, b3_7_pool, b3_9_pool])

    # Dropout later after concatenation
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)
    
    # Dense layers after concatenation+dropout
    dense = layers.Dense(64, activation = dense_activation, name = "first_dense")(cat_dropout)
    final_dense = layers.Dense(32, activation = dense_activation, name = "final_dense")(dense)
    
    # Output layer
    out = layers.Dense(1, activation = "sigmoid", name = "output_layer")(final_dense)
    
    # Model definition
    model = keras.Model(inputs = [pep, a1, a2, a3, b1, b2, b3],
                        outputs = out)
    
    return model

def CNN_CDR123_1D_baseline_embed(dropout_rate, seed, conv_activation = "relu", dense_activation = "sigmoid",
                           embed_dim = 20, nr_of_filters_1 = 32, max_lengths = None, hla_embedding_dim=66):

    # Max.length of the sequences from the dataset
    if max_lengths:
        a1_max = max_lengths[0]
        a2_max = max_lengths[1]
        a3_max = max_lengths[2]
        b1_max = max_lengths[3]
        b2_max = max_lengths[4]
        b3_max = max_lengths[5]
        pep_max = max_lengths[6]
    else:
        a1_max = 7
        a2_max = 8
        a3_max = 22
        b1_max = 6
        b2_max = 7
        b3_max = 23
        pep_max = 12

    # Input dimensions
    pep = keras.Input(shape = (pep_max, embed_dim), name ="pep")
    a1 = keras.Input(shape = (a1_max, embed_dim), name ="a1")
    a2 = keras.Input(shape = (a2_max, embed_dim), name ="a2")
    a3 = keras.Input(shape = (a3_max, embed_dim), name ="a3")
    b1 = keras.Input(shape = (b1_max, embed_dim), name ="b1")
    b2 = keras.Input(shape = (b2_max, embed_dim), name ="b2")
    b3 = keras.Input(shape = (b3_max, embed_dim), name ="b3")

    # Trained in a pan-specific setup (padding="same" for keeping original input sizes)

    # CNN layers for each feature and the different 5 kernel-sizes
    pep_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_pep_1_conv")(pep)
    pep_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_pep_3_conv")(pep)
    pep_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_pep_5_conv")(pep)
    pep_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_pep_7_conv")(pep)
    pep_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_pep_9_conv")(pep)

    a1_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a1_1_conv")(a1)
    a1_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a1_3_conv")(a1)
    a1_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a1_5_conv")(a1)
    a1_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a1_7_conv")(a1)
    a1_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a1_9_conv")(a1)

    a2_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a2_1_conv")(a2)
    a2_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a2_3_conv")(a2)
    a2_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a2_5_conv")(a2)
    a2_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a2_7_conv")(a2)
    a2_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a2_9_conv")(a2)

    a3_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a3_1_conv")(a3)
    a3_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a3_3_conv")(a3)
    a3_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a3_5_conv")(a3)
    a3_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a3_7_conv")(a3)
    a3_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a3_9_conv")(a3)

    b1_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b1_1_conv")(b1)
    b1_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b1_3_conv")(b1)
    b1_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b1_5_conv")(b1)
    b1_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b1_7_conv")(b1)
    b1_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b1_9_conv")(b1)

    b2_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b2_1_conv")(b2)
    b2_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b2_3_conv")(b2)
    b2_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b2_5_conv")(b2)
    b2_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b2_7_conv")(b2)
    b2_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b2_9_conv")(b2)

    b3_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b3_1_conv")(b3)
    b3_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b3_3_conv")(b3)
    b3_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b3_5_conv")(b3)
    b3_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b3_7_conv")(b3)
    b3_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b3_9_conv")(b3)

    # GlobalMaxPooling: takes the maximum value along the entire sequence for each feature map (16 filters)
    pep_1_pool = layers.GlobalMaxPooling1D(name = "first_pep_1_pool")(pep_1_CNN)
    pep_3_pool = layers.GlobalMaxPooling1D(name = "first_pep_3_pool")(pep_3_CNN)
    pep_5_pool = layers.GlobalMaxPooling1D(name = "first_pep_5_pool")(pep_5_CNN)
    pep_7_pool = layers.GlobalMaxPooling1D(name = "first_pep_7_pool")(pep_7_CNN)
    pep_9_pool = layers.GlobalMaxPooling1D(name = "first_pep_9_pool")(pep_9_CNN)

    a1_1_pool = layers.GlobalMaxPooling1D(name = "first_a1_1_pool")(a1_1_CNN)
    a1_3_pool = layers.GlobalMaxPooling1D(name = "first_a1_3_pool")(a1_3_CNN)
    a1_5_pool = layers.GlobalMaxPooling1D(name = "first_a1_5_pool")(a1_5_CNN)
    a1_7_pool = layers.GlobalMaxPooling1D(name = "first_a1_7_pool")(a1_7_CNN)
    a1_9_pool = layers.GlobalMaxPooling1D(name = "first_a1_9_pool")(a1_9_CNN)

    a2_1_pool = layers.GlobalMaxPooling1D(name = "first_a2_1_pool")(a2_1_CNN)
    a2_3_pool = layers.GlobalMaxPooling1D(name = "first_a2_3_pool")(a2_3_CNN)
    a2_5_pool = layers.GlobalMaxPooling1D(name = "first_a2_5_pool")(a2_5_CNN)
    a2_7_pool = layers.GlobalMaxPooling1D(name = "first_a2_7_pool")(a2_7_CNN)
    a2_9_pool = layers.GlobalMaxPooling1D(name = "first_a2_9_pool")(a2_9_CNN)

    a3_1_pool = layers.GlobalMaxPooling1D(name = "first_a3_1_pool")(a3_1_CNN)
    a3_3_pool = layers.GlobalMaxPooling1D(name = "first_a3_3_pool")(a3_3_CNN)
    a3_5_pool = layers.GlobalMaxPooling1D(name = "first_a3_5_pool")(a3_5_CNN)
    a3_7_pool = layers.GlobalMaxPooling1D(name = "first_a3_7_pool")(a3_7_CNN)
    a3_9_pool = layers.GlobalMaxPooling1D(name = "first_a3_9_pool")(a3_9_CNN)

    b1_1_pool = layers.GlobalMaxPooling1D(name = "first_b1_1_pool")(b1_1_CNN)
    b1_3_pool = layers.GlobalMaxPooling1D(name = "first_b1_3_pool")(b1_3_CNN)
    b1_5_pool = layers.GlobalMaxPooling1D(name = "first_b1_5_pool")(b1_5_CNN)
    b1_7_pool = layers.GlobalMaxPooling1D(name = "first_b1_7_pool")(b1_7_CNN)
    b1_9_pool = layers.GlobalMaxPooling1D(name = "first_b1_9_pool")(b1_9_CNN)

    b2_1_pool = layers.GlobalMaxPooling1D(name = "first_b2_1_pool")(b2_1_CNN)
    b2_3_pool = layers.GlobalMaxPooling1D(name = "first_b2_3_pool")(b2_3_CNN)
    b2_5_pool = layers.GlobalMaxPooling1D(name = "first_b2_5_pool")(b2_5_CNN)
    b2_7_pool = layers.GlobalMaxPooling1D(name = "first_b2_7_pool")(b2_7_CNN)
    b2_9_pool = layers.GlobalMaxPooling1D(name = "first_b2_9_pool")(b2_9_CNN)

    b3_1_pool = layers.GlobalMaxPooling1D(name = "first_b3_1_pool")(b3_1_CNN)
    b3_3_pool = layers.GlobalMaxPooling1D(name = "first_b3_3_pool")(b3_3_CNN)
    b3_5_pool = layers.GlobalMaxPooling1D(name = "first_b3_5_pool")(b3_5_CNN)
    b3_7_pool = layers.GlobalMaxPooling1D(name = "first_b3_7_pool")(b3_7_CNN)
    b3_9_pool = layers.GlobalMaxPooling1D(name = "first_b3_9_pool")(b3_9_CNN)

    # Concatenation of all MaxPool outputs from all features and kernel-sizes
    cat = layers.Concatenate(name = "first_cat")([pep_1_pool, pep_3_pool, pep_5_pool, pep_7_pool, pep_9_pool,
                                a1_1_pool, a1_3_pool, a1_5_pool, a1_7_pool, a1_9_pool,
                                a2_1_pool, a2_3_pool, a2_5_pool, a2_7_pool, a2_9_pool,
                                a3_1_pool, a3_3_pool, a3_5_pool, a3_7_pool, a3_9_pool,
                                b1_1_pool, b1_3_pool, b1_5_pool, b1_7_pool, b1_9_pool,
                                b2_1_pool, b2_3_pool, b2_5_pool, b2_7_pool, b2_9_pool,
                                b3_1_pool, b3_3_pool, b3_5_pool, b3_7_pool, b3_9_pool])

    # Dropout later after concatenation
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)

    embed_inp = keras.Input(shape = (hla_embedding_dim,), name ="embedding")
    final_cat = layers.Concatenate(name = "inp_cat")([cat_dropout, embed_inp])

    # Dense layers after concatenation+dropout
    dense = layers.Dense(64, activation = dense_activation, name = "first_dense")(final_cat)
    final_dense = layers.Dense(32, activation = dense_activation, name = "final_dense")(dense)

    # Output layer
    out = layers.Dense(1, activation = "sigmoid", name = "output_layer")(final_dense)

    # Model definition
    model = keras.Model(inputs = [pep, a1, a2, a3, b1, b2, b3, embed_inp],
                        outputs = out)

    return model



def CNN_CDR123_1D_baseline_embed_drop(dropout_rate, seed, conv_activation = "relu", dense_activation = "sigmoid",
                           embed_dim = 20, nr_of_filters_1 = 32, max_lengths = None, hla_embedding_dim=66):

    # Max.length of the sequences from the dataset
    if max_lengths:
        a1_max = max_lengths[0]
        a2_max = max_lengths[1]
        a3_max = max_lengths[2]
        b1_max = max_lengths[3]
        b2_max = max_lengths[4]
        b3_max = max_lengths[5]
        pep_max = max_lengths[6]
    else:
        a1_max = 7
        a2_max = 8
        a3_max = 22
        b1_max = 6
        b2_max = 7
        b3_max = 23
        pep_max = 12

    # Input dimensions
    pep = keras.Input(shape = (pep_max, embed_dim), name ="pep")
    a1 = keras.Input(shape = (a1_max, embed_dim), name ="a1")
    a2 = keras.Input(shape = (a2_max, embed_dim), name ="a2")
    a3 = keras.Input(shape = (a3_max, embed_dim), name ="a3")
    b1 = keras.Input(shape = (b1_max, embed_dim), name ="b1")
    b2 = keras.Input(shape = (b2_max, embed_dim), name ="b2")
    b3 = keras.Input(shape = (b3_max, embed_dim), name ="b3")

    # Trained in a pan-specific setup (padding="same" for keeping original input sizes)

    # CNN layers for each feature and the different 5 kernel-sizes
    pep_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_pep_1_conv")(pep)
    pep_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_pep_3_conv")(pep)
    pep_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_pep_5_conv")(pep)
    pep_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_pep_7_conv")(pep)
    pep_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_pep_9_conv")(pep)

    a1_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a1_1_conv")(a1)
    a1_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a1_3_conv")(a1)
    a1_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a1_5_conv")(a1)
    a1_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a1_7_conv")(a1)
    a1_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a1_9_conv")(a1)

    a2_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a2_1_conv")(a2)
    a2_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a2_3_conv")(a2)
    a2_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a2_5_conv")(a2)
    a2_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a2_7_conv")(a2)
    a2_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a2_9_conv")(a2)

    a3_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a3_1_conv")(a3)
    a3_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a3_3_conv")(a3)
    a3_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a3_5_conv")(a3)
    a3_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a3_7_conv")(a3)
    a3_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a3_9_conv")(a3)

    b1_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b1_1_conv")(b1)
    b1_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b1_3_conv")(b1)
    b1_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b1_5_conv")(b1)
    b1_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b1_7_conv")(b1)
    b1_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b1_9_conv")(b1)

    b2_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b2_1_conv")(b2)
    b2_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b2_3_conv")(b2)
    b2_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b2_5_conv")(b2)
    b2_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b2_7_conv")(b2)
    b2_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b2_9_conv")(b2)

    b3_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b3_1_conv")(b3)
    b3_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b3_3_conv")(b3)
    b3_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b3_5_conv")(b3)
    b3_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b3_7_conv")(b3)
    b3_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b3_9_conv")(b3)

    # GlobalMaxPooling: takes the maximum value along the entire sequence for each feature map (16 filters)
    pep_1_pool = layers.GlobalMaxPooling1D(name = "first_pep_1_pool")(pep_1_CNN)
    pep_3_pool = layers.GlobalMaxPooling1D(name = "first_pep_3_pool")(pep_3_CNN)
    pep_5_pool = layers.GlobalMaxPooling1D(name = "first_pep_5_pool")(pep_5_CNN)
    pep_7_pool = layers.GlobalMaxPooling1D(name = "first_pep_7_pool")(pep_7_CNN)
    pep_9_pool = layers.GlobalMaxPooling1D(name = "first_pep_9_pool")(pep_9_CNN)

    a1_1_pool = layers.GlobalMaxPooling1D(name = "first_a1_1_pool")(a1_1_CNN)
    a1_3_pool = layers.GlobalMaxPooling1D(name = "first_a1_3_pool")(a1_3_CNN)
    a1_5_pool = layers.GlobalMaxPooling1D(name = "first_a1_5_pool")(a1_5_CNN)
    a1_7_pool = layers.GlobalMaxPooling1D(name = "first_a1_7_pool")(a1_7_CNN)
    a1_9_pool = layers.GlobalMaxPooling1D(name = "first_a1_9_pool")(a1_9_CNN)

    a2_1_pool = layers.GlobalMaxPooling1D(name = "first_a2_1_pool")(a2_1_CNN)
    a2_3_pool = layers.GlobalMaxPooling1D(name = "first_a2_3_pool")(a2_3_CNN)
    a2_5_pool = layers.GlobalMaxPooling1D(name = "first_a2_5_pool")(a2_5_CNN)
    a2_7_pool = layers.GlobalMaxPooling1D(name = "first_a2_7_pool")(a2_7_CNN)
    a2_9_pool = layers.GlobalMaxPooling1D(name = "first_a2_9_pool")(a2_9_CNN)

    a3_1_pool = layers.GlobalMaxPooling1D(name = "first_a3_1_pool")(a3_1_CNN)
    a3_3_pool = layers.GlobalMaxPooling1D(name = "first_a3_3_pool")(a3_3_CNN)
    a3_5_pool = layers.GlobalMaxPooling1D(name = "first_a3_5_pool")(a3_5_CNN)
    a3_7_pool = layers.GlobalMaxPooling1D(name = "first_a3_7_pool")(a3_7_CNN)
    a3_9_pool = layers.GlobalMaxPooling1D(name = "first_a3_9_pool")(a3_9_CNN)

    b1_1_pool = layers.GlobalMaxPooling1D(name = "first_b1_1_pool")(b1_1_CNN)
    b1_3_pool = layers.GlobalMaxPooling1D(name = "first_b1_3_pool")(b1_3_CNN)
    b1_5_pool = layers.GlobalMaxPooling1D(name = "first_b1_5_pool")(b1_5_CNN)
    b1_7_pool = layers.GlobalMaxPooling1D(name = "first_b1_7_pool")(b1_7_CNN)
    b1_9_pool = layers.GlobalMaxPooling1D(name = "first_b1_9_pool")(b1_9_CNN)

    b2_1_pool = layers.GlobalMaxPooling1D(name = "first_b2_1_pool")(b2_1_CNN)
    b2_3_pool = layers.GlobalMaxPooling1D(name = "first_b2_3_pool")(b2_3_CNN)
    b2_5_pool = layers.GlobalMaxPooling1D(name = "first_b2_5_pool")(b2_5_CNN)
    b2_7_pool = layers.GlobalMaxPooling1D(name = "first_b2_7_pool")(b2_7_CNN)
    b2_9_pool = layers.GlobalMaxPooling1D(name = "first_b2_9_pool")(b2_9_CNN)

    b3_1_pool = layers.GlobalMaxPooling1D(name = "first_b3_1_pool")(b3_1_CNN)
    b3_3_pool = layers.GlobalMaxPooling1D(name = "first_b3_3_pool")(b3_3_CNN)
    b3_5_pool = layers.GlobalMaxPooling1D(name = "first_b3_5_pool")(b3_5_CNN)
    b3_7_pool = layers.GlobalMaxPooling1D(name = "first_b3_7_pool")(b3_7_CNN)
    b3_9_pool = layers.GlobalMaxPooling1D(name = "first_b3_9_pool")(b3_9_CNN)

    embed_inp = keras.Input(shape = (hla_embedding_dim,), name ="embedding")

    # Concatenation of all MaxPool outputs from all features and kernel-sizes, and the pep-hla embedding
    cat = layers.Concatenate(name = "first_cat")([pep_1_pool, pep_3_pool, pep_5_pool, pep_7_pool, pep_9_pool,
                                a1_1_pool, a1_3_pool, a1_5_pool, a1_7_pool, a1_9_pool,
                                a2_1_pool, a2_3_pool, a2_5_pool, a2_7_pool, a2_9_pool,
                                a3_1_pool, a3_3_pool, a3_5_pool, a3_7_pool, a3_9_pool,
                                b1_1_pool, b1_3_pool, b1_5_pool, b1_7_pool, b1_9_pool,
                                b2_1_pool, b2_3_pool, b2_5_pool, b2_7_pool, b2_9_pool,
                                b3_1_pool, b3_3_pool, b3_5_pool, b3_7_pool, b3_9_pool, embed_inp])

    # Dropout later after concatenation
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)

    # Dense layers after concatenation+dropout
    dense = layers.Dense(64, activation = dense_activation, name = "first_dense")(cat_dropout)
    final_dense = layers.Dense(32, activation = dense_activation, name = "final_dense")(dense)

    # Output layer
    out = layers.Dense(1, activation = "sigmoid", name = "output_layer")(final_dense)

    # Model definition
    model = keras.Model(inputs = [pep, a1, a2, a3, b1, b2, b3, embed_inp],
                        outputs = out)

    return model

def CNN_CDR123_1D_baseline_embed_chaintype(dropout_rate, seed, conv_activation = "relu", dense_activation = "sigmoid",
                           embed_dim = 20, nr_of_filters_1 = 32, max_lengths = None, hla_embedding_dim=66):

    # Max.length of the sequences from the dataset
    if max_lengths:
        a1_max = max_lengths[0]
        a2_max = max_lengths[1]
        a3_max = max_lengths[2]
        b1_max = max_lengths[3]
        b2_max = max_lengths[4]
        b3_max = max_lengths[5]
        pep_max = max_lengths[6]
    else:
        a1_max = 7
        a2_max = 8
        a3_max = 22
        b1_max = 6
        b2_max = 7
        b3_max = 23
        pep_max = 12

    # Input dimensions
    pep = keras.Input(shape = (pep_max, embed_dim), name ="pep")
    a1 = keras.Input(shape = (a1_max, embed_dim), name ="a1")
    a2 = keras.Input(shape = (a2_max, embed_dim), name ="a2")
    a3 = keras.Input(shape = (a3_max, embed_dim), name ="a3")
    b1 = keras.Input(shape = (b1_max, embed_dim), name ="b1")
    b2 = keras.Input(shape = (b2_max, embed_dim), name ="b2")
    b3 = keras.Input(shape = (b3_max, embed_dim), name ="b3")

    # Trained in a pan-specific setup (padding="same" for keeping original input sizes)

    # CNN layers for each feature and the different 5 kernel-sizes
    pep_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_pep_1_conv")(pep)
    pep_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_pep_3_conv")(pep)
    pep_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_pep_5_conv")(pep)
    pep_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_pep_7_conv")(pep)
    pep_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_pep_9_conv")(pep)

    a1_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a1_1_conv")(a1)
    a1_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a1_3_conv")(a1)
    a1_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a1_5_conv")(a1)
    a1_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a1_7_conv")(a1)
    a1_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a1_9_conv")(a1)

    a2_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a2_1_conv")(a2)
    a2_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a2_3_conv")(a2)
    a2_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a2_5_conv")(a2)
    a2_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a2_7_conv")(a2)
    a2_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a2_9_conv")(a2)

    a3_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_a3_1_conv")(a3)
    a3_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_a3_3_conv")(a3)
    a3_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_a3_5_conv")(a3)
    a3_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_a3_7_conv")(a3)
    a3_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_a3_9_conv")(a3)

    b1_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b1_1_conv")(b1)
    b1_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b1_3_conv")(b1)
    b1_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b1_5_conv")(b1)
    b1_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b1_7_conv")(b1)
    b1_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b1_9_conv")(b1)

    b2_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b2_1_conv")(b2)
    b2_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b2_3_conv")(b2)
    b2_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b2_5_conv")(b2)
    b2_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b2_7_conv")(b2)
    b2_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b2_9_conv")(b2)

    b3_1_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 1, padding = "same", activation = conv_activation, name = "first_b3_1_conv")(b3)
    b3_3_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 3, padding = "same", activation = conv_activation, name = "first_b3_3_conv")(b3)
    b3_5_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 5, padding = "same", activation = conv_activation, name = "first_b3_5_conv")(b3)
    b3_7_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 7, padding = "same", activation = conv_activation, name = "first_b3_7_conv")(b3)
    b3_9_CNN = layers.Conv1D(filters = nr_of_filters_1, kernel_size = 9, padding = "same", activation = conv_activation, name = "first_b3_9_conv")(b3)

    # GlobalMaxPooling: takes the maximum value along the entire sequence for each feature map (16 filters)
    pep_1_pool = layers.GlobalMaxPooling1D(name = "first_pep_1_pool")(pep_1_CNN)
    pep_3_pool = layers.GlobalMaxPooling1D(name = "first_pep_3_pool")(pep_3_CNN)
    pep_5_pool = layers.GlobalMaxPooling1D(name = "first_pep_5_pool")(pep_5_CNN)
    pep_7_pool = layers.GlobalMaxPooling1D(name = "first_pep_7_pool")(pep_7_CNN)
    pep_9_pool = layers.GlobalMaxPooling1D(name = "first_pep_9_pool")(pep_9_CNN)

    a1_1_pool = layers.GlobalMaxPooling1D(name = "first_a1_1_pool")(a1_1_CNN)
    a1_3_pool = layers.GlobalMaxPooling1D(name = "first_a1_3_pool")(a1_3_CNN)
    a1_5_pool = layers.GlobalMaxPooling1D(name = "first_a1_5_pool")(a1_5_CNN)
    a1_7_pool = layers.GlobalMaxPooling1D(name = "first_a1_7_pool")(a1_7_CNN)
    a1_9_pool = layers.GlobalMaxPooling1D(name = "first_a1_9_pool")(a1_9_CNN)

    a2_1_pool = layers.GlobalMaxPooling1D(name = "first_a2_1_pool")(a2_1_CNN)
    a2_3_pool = layers.GlobalMaxPooling1D(name = "first_a2_3_pool")(a2_3_CNN)
    a2_5_pool = layers.GlobalMaxPooling1D(name = "first_a2_5_pool")(a2_5_CNN)
    a2_7_pool = layers.GlobalMaxPooling1D(name = "first_a2_7_pool")(a2_7_CNN)
    a2_9_pool = layers.GlobalMaxPooling1D(name = "first_a2_9_pool")(a2_9_CNN)

    a3_1_pool = layers.GlobalMaxPooling1D(name = "first_a3_1_pool")(a3_1_CNN)
    a3_3_pool = layers.GlobalMaxPooling1D(name = "first_a3_3_pool")(a3_3_CNN)
    a3_5_pool = layers.GlobalMaxPooling1D(name = "first_a3_5_pool")(a3_5_CNN)
    a3_7_pool = layers.GlobalMaxPooling1D(name = "first_a3_7_pool")(a3_7_CNN)
    a3_9_pool = layers.GlobalMaxPooling1D(name = "first_a3_9_pool")(a3_9_CNN)

    b1_1_pool = layers.GlobalMaxPooling1D(name = "first_b1_1_pool")(b1_1_CNN)
    b1_3_pool = layers.GlobalMaxPooling1D(name = "first_b1_3_pool")(b1_3_CNN)
    b1_5_pool = layers.GlobalMaxPooling1D(name = "first_b1_5_pool")(b1_5_CNN)
    b1_7_pool = layers.GlobalMaxPooling1D(name = "first_b1_7_pool")(b1_7_CNN)
    b1_9_pool = layers.GlobalMaxPooling1D(name = "first_b1_9_pool")(b1_9_CNN)

    b2_1_pool = layers.GlobalMaxPooling1D(name = "first_b2_1_pool")(b2_1_CNN)
    b2_3_pool = layers.GlobalMaxPooling1D(name = "first_b2_3_pool")(b2_3_CNN)
    b2_5_pool = layers.GlobalMaxPooling1D(name = "first_b2_5_pool")(b2_5_CNN)
    b2_7_pool = layers.GlobalMaxPooling1D(name = "first_b2_7_pool")(b2_7_CNN)
    b2_9_pool = layers.GlobalMaxPooling1D(name = "first_b2_9_pool")(b2_9_CNN)

    b3_1_pool = layers.GlobalMaxPooling1D(name = "first_b3_1_pool")(b3_1_CNN)
    b3_3_pool = layers.GlobalMaxPooling1D(name = "first_b3_3_pool")(b3_3_CNN)
    b3_5_pool = layers.GlobalMaxPooling1D(name = "first_b3_5_pool")(b3_5_CNN)
    b3_7_pool = layers.GlobalMaxPooling1D(name = "first_b3_7_pool")(b3_7_CNN)
    b3_9_pool = layers.GlobalMaxPooling1D(name = "first_b3_9_pool")(b3_9_CNN)

    # Concatenation of all MaxPool outputs from all features and kernel-sizes
    cat = layers.Concatenate(name = "first_cat")([pep_1_pool, pep_3_pool, pep_5_pool, pep_7_pool, pep_9_pool,
                                a1_1_pool, a1_3_pool, a1_5_pool, a1_7_pool, a1_9_pool,
                                a2_1_pool, a2_3_pool, a2_5_pool, a2_7_pool, a2_9_pool,
                                a3_1_pool, a3_3_pool, a3_5_pool, a3_7_pool, a3_9_pool,
                                b1_1_pool, b1_3_pool, b1_5_pool, b1_7_pool, b1_9_pool,
                                b2_1_pool, b2_3_pool, b2_5_pool, b2_7_pool, b2_9_pool,
                                b3_1_pool, b3_3_pool, b3_5_pool, b3_7_pool, b3_9_pool])

    # Dropout later after concatenation
    cat_dropout = layers.Dropout(dropout_rate, seed = seed)(cat)

    embed_inp = keras.Input(shape = (hla_embedding_dim,), name ="embedding")
    chain_inp = keras.Input(shape = (2,), name ="chain_inp")

    final_cat = layers.Concatenate(name = "inp_cat")([cat_dropout, embed_inp, chain_inp])

    # Dense layers after concatenation+dropout
    dense = layers.Dense(64, activation = dense_activation, name = "first_dense")(final_cat)
    final_dense = layers.Dense(32, activation = dense_activation, name = "final_dense")(dense)

    # Output layer
    out = layers.Dense(1, activation = "sigmoid", name = "output_layer")(final_dense)

    # Model definition
    model = keras.Model(inputs = [pep, a1, a2, a3, b1, b2, b3, embed_inp, chain_inp],
                        outputs = out)

    return model


model = CNN_CDR123_1D_baseline(dropout_rate=0.5, seed=42)
model.summary()

