#!/usr/bin/python

import sys
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models, utils, optimizers, losses, regularizers
import matplotlib.pyplot as plt


# Load in data parameters
# NOTE: May need to change based on if runnning on 
# AWS or locally
X_NV_INPUT = "preprocessed/X_nv_pp.npy"
X_V_INPUT = "preprocessed/X_v_pp.npy"
Y_INPUT = "preprocessed/y_pp.npy"

## Do you want plots?
PLOT = 0

## Regularize?
REGULARIZE = 1

## RANDOM SEED TO BE FIXED THROUGHOUT
RANDOM_STATE = 1
np.random.seed(RANDOM_STATE)
tf.set_random_seed(RANDOM_STATE)

## CNN NETWORK ARCHITECTURE
'''
Inspired by LeNet and other simple/shallow CNNS while also
adding in the situational data that has some predictive power
'''

# Model HYPERPARAMETERS
# PARAMETERS TO BE INSERTED FOR TUNING
EPOCHS = int(sys.argv[1])
LEARNING_RATE = float(sys.argv[2])
MINI_BATCH = int(sys.argv[3])
NUM_UNITS = int(sys.argv[4])
FILTER_SIZE = int(sys.argv[5])

# Model input for video data
input_v = layers.Input(shape=(230, 119, 3), name='video')
input_s = layers.Input(shape=(92,), name='situation')

# CONV1
conv1 = layers.Conv2D(8, (FILTER_SIZE, FILTER_SIZE), activation='relu', 
	kernel_initializer="he_normal",
	input_shape=(230, 119, 3))(input_v)
# MAXPOOL1
maxpool1 = layers.MaxPooling2D((2, 2))(conv1)
# CONV2
conv2 = layers.Conv2D(16, (FILTER_SIZE, FILTER_SIZE), activation='relu',
	kernel_initializer="he_normal")(maxpool1)
# MAXPOOL2
maxpool2 = layers.MaxPooling2D((2, 2))(conv2)
# FLATTEN
flatten = layers.Flatten()(maxpool2)
# ADD+DENSE
concat = layers.concatenate([flatten, input_s])
if REGULARIZE:
	dense = layers.Dense(NUM_UNITS, activation="relu",
		kernel_initializer="he_normal",
		kernel_regularizer=regularizers.l2(0.001))(concat)
else:
	dense = layers.Dense(NUM_UNITS, activation="relu",
		kernel_initializer="he_normal")(concat)

# OUTPUT
if REGULARIZE:
	out = layers.Dense(1, activation="linear", 
		kernel_regularizer=regularizers.l2(0.001))(dense)
else:
	out = layers.Dense(1, activation="linear")(dense)

# Define model
model = models.Model(inputs=[input_v, input_s], outputs=out)

# Define optimizer and metrics
model.compile(optimizer=optimizers.Adam(lr=LEARNING_RATE),
              metrics=["mean_absolute_error"],
              loss="mean_absolute_error")


#### RUN CODE
# Now LOAD IN DATA AND TRAIN
# Data augment - flip around the images (so double the amount)
	# This should be easy to do as a first step




# Train/Dev/Test split
TRAIN_PER = 0.75
DEV_PER = 0.125
TEST_PER = 0.125
RANDOM_STATE = 1

# Which y variable to fit over
WHICH_Y = 0

# Data augment?
DATA_AUGMENT = 0


##### START CODE

# Load in data
X_nv = np.load(X_NV_INPUT, allow_pickle=True)
X_v = np.load(X_V_INPUT, allow_pickle=True) / 255.0
Y = np.load(Y_INPUT, allow_pickle=True).astype("float64")

# Split into training, dev and test
m = X_nv.shape[0]
shuffled_indices = np.arange(m)
np.random.shuffle(shuffled_indices)

train_idxs = shuffled_indices[:int(m*TRAIN_PER)]
dev_idxs = shuffled_indices[(int(m*TRAIN_PER)):(int(m*TRAIN_PER)+int(m*DEV_PER))]
test_idxs = shuffled_indices[(int(m*TRAIN_PER)+int(m*DEV_PER)):]

X_v_train, X_nv_train, Y_train = X_v[train_idxs,:,:,:], X_nv[train_idxs,:], Y[train_idxs, WHICH_Y]
X_v_dev, X_nv_dev, Y_dev = X_v[dev_idxs,:,:,:], X_nv[dev_idxs,:], Y[dev_idxs, WHICH_Y]
X_v_test, X_nv_test, Y_test= X_v[test_idxs,:,:,:], X_nv[test_idxs,:], Y[test_idxs, WHICH_Y]

# DATA AUGMENT
if DATA_AUGMENT:
	None


# Training
history = model.fit(
	{'video' : X_v_train,
	'situation' : X_nv_train},
                    Y_train,
                    epochs=EPOCHS,
                    verbose=2,
                    batch_size=MINI_BATCH)



#loss_history = history.history['loss']
#val_loss_history = history.history['val_loss']

if PLOT:
	plt.plot(history.epoch,
		loss_history,
		history.epoch)
	plt.ylabel('cost')
	plt.xlabel('epoch')
	plt.show()

# Evaluating model
fits = model.predict({'video' : X_v_dev, 'situation' : X_nv_dev})
final_score_SCNN = np.median(np.abs(np.around(fits)- Y_dev))
print(fits[0:99].astype(int))
print("Epochs: "+str(EPOCHS))
print("Learning Rate: "+str(LEARNING_RATE))
print("Mini Batch: "+str(MINI_BATCH))
print("Num Hidden Units: "+str(NUM_UNITS))
print("Filter Size: "+str(FILTER_SIZE))
print("Eval Metric on Dev Set: "+str(final_score_SCNN))
test_fits = model.predict({'video' : X_v_test, 'situation' : X_nv_test})
final_score_test = np.median(np.abs(np.around(test_fits)- Y_test))
print("Eval Metric on Test Set: "+str(final_score_test))
print("Mean abs yards on Dev: "+str(np.mean(np.abs(np.around(fits)- Y_dev))))
print("SD of predicted yards: "+str(np.std(test_fits.astype(int))))
print("Mean abs yards on Test: "+str(np.mean(np.abs(np.around(test_fits)- Y_test))))
