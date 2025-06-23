# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
@NetTCR2.2_author: Mathias
@modifications_authors: Carlos, Jonas
"""

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score

import os
import sys
import numpy as np
import pandas as pd

# Imports the util module and network architectures for NetTCR
# If changes are wanted, change this directory to your own directory
# Directory with the "keras_utils.py" script and model architecture
sys.path.append("/home/projects2/jonnil/nettcr_2024/nettcr_2d/NetTCR-2Dmaps/keras_src")

import keras_utils
import matplotlib.pyplot as plt
import seaborn as sns
import random
import argparse

# Importing the model
from CNN_keras_carlos_architecture import CNN_CDR123_1D_baseline

# Makes the plots look better
sns.set()

def args_parser():
    parser = argparse.ArgumentParser(description='NetTCR training script')
    """
    Data processing args
    """
    parser.add_argument('-trf', '--train_file', dest='train_file', required=True, type=str,
                        default='/home/projects/vaccine/people/cadsal/MasterThesis/data/nettcr_2_2_full_dataset.csv',
                        help='Filename of the data input file')
    parser.add_argument('-out', '--out_dir', dest='out_directory', required=True, type=str,
                        default='/home/projects/vaccine/people/cadsal/MasterThesis/baseline/outdir',
                        help='Output directory')
    parser.add_argument("-tp", "--test_partition", dest='test_fold', required=False, type=int,
                        default=0, help='Test partition for the nested-CV')
    parser.add_argument("-vp", "--valid_partition", dest='valid_fold', required=False, type=int,
                        default=1, help='Test partition for the nested-CV')
    parser.add_argument("-s", "--seed", dest="seed", required=False, type=int,
                        default=1, help='Seed for fixing random initializations')
    parser.add_argument("-do", "--dropout", dest="dropout", required=False, type=float,
                        default=0.6, help='Dropout value for CNN layers after max.pooling and concatenating them')
    parser.add_argument("-lr", "--learning_rate", dest="learning_rate", required=False, type=float,
                        default=0.0005, help='Learning rate')
    parser.add_argument("-w", "--sample_weight", dest="sample_weight", action="store_true",
                        default=False, help='Use sample weighting')
    return parser.parse_args()

# Parse the command-line arguments and store them in the 'args' variable
args = args_parser()
args_dict = vars(args)

# Define the test and validation partitions
t = int(args_dict['test_fold'])
v = int(args_dict['valid_fold'])

# Set random seed
seed = int(args.seed)
np.random.seed(seed)      # Numpy module
random.seed(seed)         # Python random module
tf.random.set_seed(seed)  # Tensorflow random seed

# Plots function
def plot_loss_auc(train_losses, valid_losses, train_aucs, valid_aucs,
                   filename, dpi=300):
    f, a = plt.subplots(2, 1, figsize=(12, 10))
    a = a.ravel()
    a[0].plot(train_losses, label='train_losses')
    a[0].plot(valid_losses, label='valid_losses')
    a[0].legend()
    a[0].set_title('Baseline CDR123 Loss (TestFold: ' + str(t) + ', ValidFold: ' + str(v) + ')')
    a[1].plot(train_aucs, label='train_aucs')
    a[1].plot(valid_aucs, label='valid_aucs')
    a[1].legend()
    a[1].set_title('Baseline CDR123 AUC (TestFold: ' + str(t) + ', ValidFold: ' + str(v) + ')')
    a[1].set_xlabel('Epoch')
    f.savefig(f'{filename}', dpi=dpi, bbox_inches='tight')

def plot_loss_auc01(train_losses, valid_losses, train_aucs, valid_aucs,
                    filename, dpi=300):
    f, a = plt.subplots(2, 1, figsize=(12, 10))
    a = a.ravel()
    a[0].plot(train_losses, label='train_losses')
    a[0].plot(valid_losses, label='valid_losses')
    a[0].legend()
    a[0].set_title('Baseline CDR123 Loss (TestFold: ' + str(t) + ', ValidFold: ' + str(v) + ')')
    a[1].plot(train_aucs, label='train_aucs')
    a[1].plot(valid_aucs, label='valid_aucs')
    a[1].legend()
    a[1].set_title('Baseline CDR123 AUC0.1 (TestFold: ' + str(t) + ', ValidFold: ' + str(v) + ')')
    a[1].set_xlabel('Epoch')
    f.savefig(f'{filename}', dpi=dpi, bbox_inches='tight')

### Input/Output ###

# Read in data
data = pd.read_csv(args_dict['train_file'])
# Directories
outdir = args_dict['out_directory']

### Weigthed loss definition ###

if args_dict['sample_weight']:

    # Sample weights
    weight_dict = np.log2(data.shape[0]/(data.peptide.value_counts()))
    # Normalize, so that loss is comparable
    weight_dict = weight_dict*(data.shape[0]/np.sum(weight_dict*data.peptide.value_counts()))
    data["sample_weight"] = data["peptide"].map(weight_dict)
    # # Adjust according to if the observation include both paired-chain sequence data for the CDRs
    # weight_multiplier_dict = {"alpha": 1, "beta": 1, "paired": 2}
    # data["weight_multiplier"] = data.input_type.map(weight_multiplier_dict)
    # data["sample_weight"] = data["sample_weight"]*data["weight_multiplier"]/weight_multiplier_dict["paired"]

else:
    # If sample_weight is False, set sample_weight to 1 for all rows
    data["sample_weight"] = 1

# Define the list of binding peptides in the data (descending order according to the number of occurrences)
pep_list = list(data[data.binder==1].peptide.value_counts(ascending=False).index)

### Model training parameters ###

train_parts = {0, 1, 2, 3, 4}                    # Partitions
patience = 50                                    # Patience for Early Stopping
dropout_rate = args_dict['dropout']              # Dropout Rate
encoding = keras_utils.blosum50_20aa_masking     # Encoding for amino acid sequences
EPOCHS = 200                                     # Number of epochs in the training
batch_size = 64                                  # Number of elements in each batch

# Padding to max. length according to the observations in the dataset
a1_max = 7
a2_max = 8
a3_max = 22
b1_max = 6
b2_max = 7
b3_max = 23
pep_max = 12

def make_tf_ds(df, encoding):
    """Prepares the embedding for the input features to the model"""
    # Normalization factor of 5
    encoded_pep = keras_utils.enc_list_bl_max_len(df.peptide, encoding, pep_max)/5
    encoded_a1 = keras_utils.enc_list_bl_max_len(df.A1, encoding, a1_max)/5
    encoded_a2 = keras_utils.enc_list_bl_max_len(df.A2, encoding, a2_max)/5
    encoded_a3 = keras_utils.enc_list_bl_max_len(df.A3, encoding, a3_max)/5
    encoded_b1 = keras_utils.enc_list_bl_max_len(df.B1, encoding, b1_max)/5
    encoded_b2 = keras_utils.enc_list_bl_max_len(df.B2, encoding, b2_max)/5
    encoded_b3 = keras_utils.enc_list_bl_max_len(df.B3, encoding, b3_max)/5
    targets = df.binder.values
    sample_weights = df.sample_weight
    tf_ds = [encoded_pep,
             encoded_a1, encoded_a2, encoded_a3,
             encoded_b1, encoded_b2, encoded_b3,
             targets,
             sample_weights]

    return tf_ds

# AUC custom function
def my_numpy_function(y_true, y_pred):
    try:
        auc = roc_auc_score(y_true, y_pred, max_fpr = 0.1)
    except ValueError:
        # Exception for when a positive observation is not present in a batch
        auc = np.array([float(0)])
    return auc

# Custom metric for AUC 0.1
def auc_01(y_true, y_pred):
    "Allows Tensorflow to use the function during training"
    auc_01 = tf.numpy_function(my_numpy_function, [y_true, y_pred], tf.float64)
    return auc_01

# Creates the directory to save the model in
if not os.path.exists(outdir):
    os.makedirs(outdir)

dependencies = {
    'auc_01': auc_01
}

outfile = open(outdir + "/" + "s.{}.t.{}.v.{}.fold_validation.tsv".format(seed,t,v), mode = "w")
print("fold", "valid_loss_auc01", "best_auc_0.1", "best_epoch_auc01", "best_auc", "best_epoch_auc", sep = "\t", file = outfile)

# Prepare plotting
fig, ax = plt.subplots(figsize=(15, 10))

### Data and model initialization ###

# Training data (not including validation and test sets)
x_train_df = data[(data.partition!=t)&(data.partition!=v)].reset_index()
train_tensor = make_tf_ds(x_train_df, encoding = encoding)
x_train = train_tensor[0:7]      # Selecting inputs
targets_train = train_tensor[7]  # Target (0 or 1)
weights_train = train_tensor[8]  # Sample weight for the loss function

# Validation data - Used for early stopping
# x_valid_df = data[(data.partition==v) & (data.input_type == "paired")]
x_valid_df = data[(data.partition==v)]
valid_tensor = make_tf_ds(x_valid_df, encoding = encoding)
x_valid = valid_tensor[0:7]      # Inputs
targets_valid = valid_tensor[7]  # Target (0 or 1)
weights_valid = valid_tensor[8]  # Sample weight for the loss function

# Selection of the model to train
model = CNN_CDR123_1D_baseline(dropout_rate, seed, nr_of_filters_1 = 32)

# Saves the model at the best epoch (based on validation loss or other metric)
ModelCheckpoint = keras.callbacks.ModelCheckpoint(
        filepath = outdir + '/checkpoint/' + 's.' + str(seed) + '.t.' + str(t) + '.v.' + str(v) + ".keras",
        monitor = "val_auc_01",
        mode = "max",
        save_best_only = True)

# EarlyStopping function used for stopping model training when the model does not improve
EarlyStopping = keras.callbacks.EarlyStopping(
    monitor = "val_auc_01",
    mode = "max",
    patience = patience)

# Callbacks to include for the model training
callbacks_list = [EarlyStopping,
                  ModelCheckpoint
    ]

# Optimizers, loss functions, and additional metrics to track
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = args_dict['learning_rate']),
              loss = tf.keras.losses.BinaryCrossentropy(),
              metrics = [auc_01, "AUC"],
              weighted_metrics = [])

# Print the summary of the model
#print("")
#model.summary()

### Announce Training ###

print("Training model with test_partition = {} & validation_partition = {}".format(t,v), end = "\n")


print("x_train[0].shape", x_train[0].shape)
#sys.exit()

# Model training
history = model.fit(x = {"pep": x_train[0],
                         "a1": x_train[1],
                         "a2": x_train[2],
                         "a3": x_train[3],
                         "b1": x_train[4],
                         "b2": x_train[5],
                         "b3": x_train[6]},
          y = targets_train,
          batch_size = batch_size,
          epochs = EPOCHS,
          verbose = 2,
          sample_weight = weights_train,
          validation_data = ({"pep": x_valid[0],
                              "a1": x_valid[1],
                              "a2": x_valid[2],
                              "a3": x_valid[3],
                              "b1": x_valid[4],
                              "b2": x_valid[5],
                              "b3": x_valid[6]},
                             targets_valid,
                             weights_valid),
          validation_batch_size = batch_size,
          shuffle = True,
          callbacks=callbacks_list
          )

# Loss and metrics for each epoch during training
valid_loss = history.history["val_loss"]
train_loss = history.history["loss"]
valid_auc = history.history["val_auc"]
valid_auc01 = history.history["val_auc_01"]
train_auc = history.history["auc"]
train_auc01 = history.history["auc_01"]

plotdir = outdir + "/plots"

if not os.path.exists(plotdir):
    os.makedirs(plotdir)

# Plotting the losses
ax.plot(train_loss, label='train')
ax.plot(valid_loss, label='validation')
ax.set_ylabel("Loss")
ax.set_xlabel("Epoch")
ax.legend()
ax.set_title('Baseline CDR123 Loss (TestFold: ' + str(t) + ', ValidFold: ' + str(v) + ')')
# Save training/validation loss plot
plt.tight_layout()
plt.show()
fig.savefig(plotdir + '/s.{}.t.{}.v.{}.learning_curves.png'.format(seed,t,v), dpi=200)

# Plotting the losses with the AUC value
plot_loss_auc(train_loss, valid_loss, train_auc, valid_auc,
              plotdir + '/s.{}.t.{}.v.{}.lossVSAUC.png'.format(seed,t,v), dpi=300)
plot_loss_auc01(train_loss, valid_loss, train_auc01, valid_auc01,
                plotdir + '/s.{}.t.{}.v.{}.lossVSAUC01.png'.format(seed,t,v), dpi=300)


# Record metrics at checkpoint
fold = outdir + '/checkpoint/' + "s." + str(seed) + '.t.' + str(t) + '.v.' + str(v) + ".keras"
valid_best = valid_loss[np.argmax(valid_auc01)]
best_epoch_auc01 = np.argmax(valid_auc01)
best_auc01 = np.max(valid_auc01)
best_epoch_auc = np.argmax(valid_auc)
best_auc = np.max(valid_auc)

# Load the best model
model = keras.models.load_model(outdir + '/checkpoint/' + 's.' + str(seed) +  '.t.' + str(t) + '.v.' + str(v) + ".keras", custom_objects=dependencies)

# Converting the model to a TFlite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# Save the model
with open(outdir + '/checkpoint/' + 's.' + str(seed) +  '.t.' + str(t) + '.v.' + str(v) + ".tflite", 'wb') as f:
  f.write(tflite_model)

# Records loss and metrics at saved epoch
print(fold, valid_best, best_auc01, best_epoch_auc01, best_auc, best_epoch_auc, sep = "\t", file = outfile)

# Clears the session for the next model
tf.keras.backend.clear_session()

# Close log file
outfile.close()

