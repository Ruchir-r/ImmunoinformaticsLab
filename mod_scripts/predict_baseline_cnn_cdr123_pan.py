# -*- coding: utf-8 -*-
"""
@NetTCR2.2_author: Mathias
@modifications_authors: Carlos, Jonas
"""

#!/usr/bin/env python

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score

import os
import sys
import numpy as np
import pandas as pd

#Imports the util module and network architectures for NetTCR
#If changes are wanted, change this directory to your own directory
#sys.path.append("/home/projects/vaccine/people/matjen/master_project/nettcr_src")

#Directory with the "keras_utils.py" script
sys.path.append("/home/projects2/jonnil/nettcr_2024/nettcr_2d/NetTCR-2Dmaps/keras_src")

import keras_utils
import random
import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='NetTCR training script')
    """
    Data processing args
    """
    parser.add_argument('-f', '--data_file', dest='data_file', required=True, type=str,
                        default='/home/projects/vaccine/people/cadsal/MasterThesis/data/nettcr_2_2_full_dataset.csv',
                        help='Filename of the data input file')
    parser.add_argument('-out', '--out_dir', dest='out_directory', required=True, type=str,
                        default='/home/projects/vaccine/people/cadsal/MasterThesis/baseline/outdir',
                        help='Filename of the data input file')
    parser.add_argument("-s", "--seeds", dest="seeds", required=False, nargs="+", type=int,
                        default=[1], help='Seeds used for models')
    return parser.parse_args()

# Parse the command-line arguments and store them in the 'args' variable
args = args_parser()
args_dict = vars(args)


### Input/Output ###

# Read in data
data = pd.read_csv(args_dict['data_file'])

# Directories
outdir = args_dict['out_directory']
predict_dir = outdir + '/predict'

# Define the list of peptides in the training data
pep_list = list(data[data.binder==1].peptide.value_counts(ascending=False).index)

### Model training parameters ###

train_parts = {0, 1, 2, 3, 4}                 # Partitions
encoding = keras_utils.blosum50_20aa_masking  # Encoding for amino acid sequences

# Padding to certain length
a1_max = 7
a2_max = 8
a3_max = 22
b1_max = 6
b2_max = 7
b3_max = 23
pep_max = 12

# AUC0.1 custom function
def my_numpy_function(y_true, y_pred):
    try:
        auc = roc_auc_score(y_true, y_pred, max_fpr = 0.1)
    except ValueError:
        #Exception for when a positive observation is not present in a batch
        auc = np.array([float(0)])
    return auc

# Custom metric for AUC 0.1
def auc_01(y_true, y_pred):
    "Allows Tensorflow to use the function during training"
    auc_01 = tf.numpy_function(my_numpy_function, [y_true, y_pred], tf.float64)
    return auc_01

# Necessary to load the model with the custom metric
dependencies = {
    'auc_01': auc_01
}

def make_tf_ds(df, encoding):
    """Prepares the embedding for the input features to the model"""
    encoded_pep = keras_utils.enc_list_bl_max_len(df.peptide, encoding, pep_max)/5
    encoded_a1 = keras_utils.enc_list_bl_max_len(df.A1, encoding, a1_max)/5
    encoded_a2 = keras_utils.enc_list_bl_max_len(df.A2, encoding, a2_max)/5
    encoded_a3 = keras_utils.enc_list_bl_max_len(df.A3, encoding, a3_max)/5
    encoded_b1 = keras_utils.enc_list_bl_max_len(df.B1, encoding, b1_max)/5
    encoded_b2 = keras_utils.enc_list_bl_max_len(df.B2, encoding, b2_max)/5
    encoded_b3 = keras_utils.enc_list_bl_max_len(df.B3, encoding, b3_max)/5
    targets = df.binder.values
    tf_ds = [np.float32(encoded_pep),
             np.float32(encoded_a1), np.float32(encoded_a2), np.float32(encoded_a3),
             np.float32(encoded_b1), np.float32(encoded_b2), np.float32(encoded_b3),
             targets]

    return tf_ds

# Creates the directory to save the test predictions
if not os.path.exists(predict_dir):
    os.makedirs(predict_dir)

# Prepare output dataframe (test predictions)
pred_df = pd.DataFrame()

# Loop over each model by excluding the test partition defined in each CV round
for t in train_parts:

    x_test_df = data[(data.partition==t)].reset_index()
    test_tensor = make_tf_ds(x_test_df, encoding = encoding)
    x_test = test_tensor[0:7]          # Features
    targets_test = test_tensor[7]      # Target values
    avg_prediction = 0                 # Reset prediction

    # Loop over each validation fold that is not the same of the test fold
    for v in train_parts:

        if v!=t:

            for s in args.seeds:

                # Load the TFLite model and allocate tensors
                interpreter = tf.lite.Interpreter(model_path = outdir + '/checkpoint/s.{}.t.{}.v.{}.tflite'.format(s, t, v))

                # Get input and output tensors for the model
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                # Fix Output dimensions
                output_shape = output_details[0]['shape']
                interpreter.resize_tensor_input(output_details[0]["index"], [x_test[0].shape[0], output_details[0]["shape"][1]])

                # Fix Input dimensions
                for i in range(len(input_details)):
                    interpreter.resize_tensor_input(input_details[i]["index"], [x_test[0].shape[0], input_details[i]["shape"][1], input_details[i]["shape"][2]])

                # Prepare tensors
                interpreter.allocate_tensors()

                data_dict = {"pep": x_test[0],
                             "a1": x_test[1],
                             "a2": x_test[2],
                             "a3": x_test[3],
                             "b1": x_test[4],
                             "b2": x_test[5],
                             "b3": x_test[6]}

                # Assign input data
                for i in range(len(input_details)):
                    # Set input data for a given feature based on the name of the input in "input_details"
                    interpreter.set_tensor(input_details[i]['index'], data_dict[input_details[i]["name"].split(":")[0].split("_")[-1]])


                # Ready the model
                interpreter.invoke()

                # The function `get_tensor()` returns a copy of the tensor data
                # Use `tensor()` in order to get a pointer to the tensor
                avg_prediction += interpreter.get_tensor(output_details[0]['index'])

                # Clears the session for the next model
                tf.keras.backend.clear_session()

    # Averaging the predictions between all models in the inner loop
    avg_prediction = avg_prediction/4
    x_test_df['prediction'] = avg_prediction
    pred_df = pd.concat([pred_df, x_test_df])

# Save predictions
pred_df.to_csv(predict_dir + '/testCV_pred_df.csv', index=False)

