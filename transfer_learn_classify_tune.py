#!/usr/bin/python

import sys
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tensorflow.keras import datasets, layers, models, utils, optimizers, losses, regularizers
import matplotlib.pyplot as plt
from itertools import product


## RANDOM SEED TO BE FIXED THROUGHOUT
RANDOM_STATE = 1
np.random.seed(RANDOM_STATE)
tf.set_random_seed(RANDOM_STATE)

## PARAMETERS
# Dev score output
DEV_SCORE_OUTPUT = "dev_score_transfer_classify.csv"
headers = "dev_score,epochs,learning_rate,mini_batches,num_hidden_units,num_base_units,\n"
f = open(DEV_SCORE_OUTPUT, "w")
f.write(headers)


# Load in data parameters
# NOTE: May need to change based on if runnning on 
# AWS or locally
X_NV_INPUT = "preprocessed/X_nv_pp.npy"
X_V_INPUT = "preprocessed/X_v_pp.npy"
Y_INPUT = "preprocessed/y_pp.npy"

# HYPERPARAMETERS setup
epochs = np.array([20])
r_learning_rate = -4 * np.random.rand(25)
hp_grid_learn_rate = np.sort(10**r_learning_rate)
mini_batches = np.array([16, 32])
num_units = np.array([2,3,4])
num_base_units = np.array([2,3,4,5])

hp_grid = np.array(list(product(epochs, hp_grid_learn_rate, mini_batches, num_units, num_base_units)))


# DATA
# Load in data
X_nv = np.load(X_NV_INPUT, allow_pickle=True)
X_v = np.load(X_V_INPUT, allow_pickle=True) / 255.0
Y = np.load(Y_INPUT, allow_pickle=True).astype("float64")

# Train/Dev/Test split
TRAIN_PER = 0.75
DEV_PER = 0.125
TEST_PER = 0.125
RANDOM_STATE = 1

# Which y variable to fit over
WHICH_Y = 1

# Data augment?
DATA_AUGMENT = 0


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


## MODEL SETUP + TRAINING
input_v = layers.Input(shape=(230, 119, 3), name='video')
input_s = layers.Input(shape=(92,), name='situation')

# Specify base model

img_shape = (230, 119, 3)
base_model = tf.keras.applications.vgg19.VGG19(input_shape=img_shape,
                                               include_top=False,
                                               weights='imagenet')

# Over the hyperparameters
for i in range(hp_grid.shape[0]):
	EPOCHS = int(hp_grid[i,0])
	LEARNING_RATE = float(hp_grid[i,1])
	MINI_BATCH = int(hp_grid[i,2])
	NUM_UNITS = int(hp_grid[i,3])
	NUM_BASE_LAYERS = int(hp_grid[i,4])
	# Model input for image and situation data
	
	# Get earlier layers from VGG19
	# Idea: Only want low level features
	# Will mess around with this
	low_name = "block"+str(NUM_BASE_LAYERS)+"_pool"

	base_model_low_out = base_model.get_layer(low_name).output
	base_model_low = models.Model(base_model.input, base_model_low_out)
	# Freeze all layers
	base_model_low.trainable = False
	base_low_out = base_model_low(input_v)
	# Testing to see if it matches lower level features
	#low_level_feature_model = models.Model(inputs=base_model.input, 
	#	outputs=base_model.get_layer('block3_pool').output)
	#low_level_output = low_level_feature_model.predict(X_v_train[1,].reshape(1,230,119,3))


	# Flatten base model
	flatten = layers.Flatten()(base_low_out)

	# Add in situational data
	concat = layers.concatenate([flatten, input_s])

	# Pass through dense layer
	dense = layers.Dense(NUM_UNITS, activation="relu", 
		kernel_initializer="he_normal")(concat)

	# Output layer for binary classification
	out = layers.Dense(1, activation="sigmoid")(dense)

	# Define model
	model = models.Model(inputs=[input_v, input_s], outputs=out)

	# Define optimizer and metrics
	model.compile(optimizer=optimizers.Adam(lr=LEARNING_RATE),
		metrics=["accuracy"],
		loss='binary_crossentropy')

	# Intermediate feature
	#intermediate_layer_model = models.Model(inputs=base_model.input,
	#                                 outputs=model.get_layer("block3_pool").output)
	#intermediate_output = intermediate_layer_model.predict(X_v_train[1,].reshape(1,230,119,3))

	## MODEL TRAINING

	# Fit model
	history = model.fit(
		{'video' : X_v_train,
		'situation' : X_nv_train},
        	            Y_train,
            	        epochs=EPOCHS,
                	    verbose=2,
                    	batch_size=MINI_BATCH)

	# Evaluate model
	fits = 1.0*(model.predict({'video' : X_v_dev, 'situation' : X_nv_dev})>=0.5)
	accuracy_dev_score = accuracy_score(Y_dev, fits)
	f.write(str(accuracy_dev_score) + ',' + str(EPOCHS) + ',' + str(LEARNING_RATE) + ',' + str(MINI_BATCH) + ',' + str(NUM_UNITS) + ',' + str(NUM_BASE_LAYERS) + '\n')
	print("Accuracy on Dev Set: "+str(accuracy_dev_score))
	print('\n')
	print('\n')
	print(str(i)+' / '+str(hp_grid.shape[0]))
	print('\n')
	print('\n')

f.close()
