#!/usr/bin/python

import sys
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tensorflow.keras import datasets, layers, models, utils, optimizers, losses, regularizers
import matplotlib.pyplot as plt
from PIL import Image
import keras_vis


## RANDOM SEED TO BE FIXED THROUGHOUT
RANDOM_STATE = 1
np.random.seed(RANDOM_STATE)
tf.set_random_seed(RANDOM_STATE)

## PARAMETERS
# Load in data parameters
# NOTE: May need to change based on if runnning on 
# AWS or locally
X_NV_INPUT = "preprocessed/X_nv_pp.npy"
X_V_INPUT = "preprocessed/X_v_pp.npy"
Y_INPUT = "preprocessed/y_pp.npy"

# HYPERPARAMETERS
EPOCHS = int(sys.argv[1])
LEARNING_RATE = float(sys.argv[2])
MINI_BATCH = int(sys.argv[3])
NUM_UNITS = int(sys.argv[4])
NUM_BASE_LAYERS = int(sys.argv[5])



## MODEL SETUP

# Model input for image and situation data
input_v = layers.Input(shape=(230, 119, 3), name='video')
input_s = layers.Input(shape=(92,), name='situation')

# Specify base model

img_shape = (230, 119, 3)
base_model = tf.keras.applications.vgg19.VGG19(input_shape=img_shape,
                                               include_top=False,
                                               weights='imagenet')

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
#concat = layers.concatenate([flatten, input_s])
concat = flatten

# Pass through dense layer
dense = layers.Dense(NUM_UNITS, activation="relu", 
	kernel_initializer="he_normal")(concat)

# Output layer for 
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
# Train/Dev/Test split
TRAIN_PER = 0.75
DEV_PER = 0.125
TEST_PER = 0.125
RANDOM_STATE = 1

# Which y variable to fit over
WHICH_Y = 1

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

# Fit model
history = model.fit(
	{'video' : X_v_train,
	'situation' : X_nv_train},
                    Y_train,
                    epochs=EPOCHS,
                    verbose=2,
                    batch_size=MINI_BATCH,
                    validation_data = ({'video' : X_v_dev, 'situation' : X_nv_dev}, Y_dev))

# Evaluate model
fits = 1.0*(model.predict({'video' : X_v_dev, 'situation' : X_nv_dev})>=0.5).reshape(len(dev_idxs),)
accuracy_dev_score = accuracy_score(Y_dev, fits)
print(fits[np.where(fits!=Y_dev)])
print(Y_dev[np.where(fits!=Y_dev)])
print(dev_idxs[np.where(fits!=Y_dev)])
print("Epochs: "+str(EPOCHS))
print("Learning Rate: "+str(LEARNING_RATE))
print("Mini Batch: "+str(MINI_BATCH))
print("Num Hidden Units: "+str(NUM_UNITS))
print("Num Base layers from VGG: "+str(NUM_BASE_LAYERS))
print("Accuracy on Dev Set: "+str(accuracy_dev_score))
test_fits = 1.0*(model.predict({'video' : X_v_test, 'situation' : X_nv_test})>=0.5)
accuracy_test_score = accuracy_score(Y_test, test_fits)
print("Accuracy on Test Set: "+str(accuracy_test_score))
