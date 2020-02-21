import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt



## CNN NETWORK ARCHITECTURE
'''
Inspired by LeNet and other simple/shallow CNNS while also
adding in the situational data that has some predictive power
'''

# Model HYPERPARAMETERS


# Initiate model
model = models.Sequential()

# CONV1 Layer
model.add(layers.Conv2D(8, (5, 5), 
	activation='relu', 
	input_shape=(230, 119, 3)))

# MAXPOOL1 Layer
model.add(layers.MaxPooling2D((2, 2)))

# CONV2 Layer
model.add(layers.Conv2D(16, (5, 5), 
	activation='relu'))

# MAXPOOL2 Layer
model.add(layers.MaxPooling2D((2, 2)))

# CONV3 Layer
model.add(layers.Conv2D(16, (5, 5), 
	activation='relu'))

# Flatten layer
model.add(layers.Flatten())

# Penultimate dense layer with situational variables added
model.add(layers.Dense(5, 
	activation="relu",
	input_shape=(37536+4,)))

# Final dense layer
# NOTE: Linear activation because regression output
model.add(layers.Dense(1, 
	activation="linear"))


# Model compilation
model.compile(optimizer="adam",
              loss="mean_squared_error",
              metrics=["mean_squared_error",
              "mean_absolute_error"])


#### RUN CODE
# Now LOAD IN DATA AND TRAIN
# Data augment - flip around the images (so double the amount)
	# This should be easy to do as a first step


# Load in data
X_NV_INPUT = "data/preprocessed/X_nv_pp.npy"
X_V_INPUT = "data/preprocessed/X_v_pp.npy"
Y_INPUT = "data/preprocessed/y_pp.npy"

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


X_v_train, X_v_test, Y_train, Y_test = train_test_split(X_v, Y, test_size=TEST_PER, random_state=RANDOM_STATE)
X_v_train, X_v_dev, Y_train, Y_dev = train_test_split(X_v_train, Y_train, test_size=DEV_PER/(1-TEST_PER), random_state=RANDOM_STATE)

# DATA AUGMENT
if DATA_AUGMENT:
	None

# Training 
history = model.fit(X_v_train,
                    Y_train,
                    epochs=10,
                    verbose=2,
                    batch_size=32,
                    validation_data=(X_v_dev, Y_dev))

loss_history = history.history['loss']
val_loss_history = history.history['val_loss']

plt.plot(history.epoch,
	loss_history)
plt.ylabel('cost')
plt.ylabel('epoch')
plt.show()

# Evaluating model
model.evaluate(X_v_test, Y_test, verbose=2)
fits = model.predict(X_v_test)
final_score_SCNN = np.mean(np.abs(fits- Y_test[:, WHICH_Y]))
final_score_SCNN2 = np.median(np.abs(fits- Y_test[:, WHICH_Y]))

