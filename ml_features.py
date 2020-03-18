import numpy as np
import sklearn
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score
from itertools import product

## PARAMETERS
# Input data
X_NV_INPUT = "data/preprocessed/X_nv_pp.npy"
Y_INPUT = "data/preprocessed/y_pp.npy"

# Train/Dev/Test split
TRAIN_PER = 0.75
DEV_PER = 0.125
TEST_PER = 0.125
RANDOM_STATE = 1

# Which y variable to fit over
WHICH_Y = 0


##### START CODE

# Load in data
X_nv = np.load(X_NV_INPUT, allow_pickle=True)
Y = np.load(Y_INPUT, allow_pickle=True).astype("float64")



# Split data into train, dev, test
np.random.seed(RANDOM_STATE)
m = X_nv.shape[0]
shuffled_indices = np.arange(m)
np.random.shuffle(shuffled_indices)

train_idxs = shuffled_indices[:int(m*TRAIN_PER)]
dev_idxs = shuffled_indices[(int(m*TRAIN_PER)):(int(m*TRAIN_PER)+int(m*DEV_PER))]
test_idxs = shuffled_indices[(int(m*TRAIN_PER)+int(m*DEV_PER)):]

X_train, Y_train = X_nv[train_idxs,:], Y[train_idxs, WHICH_Y]
X_dev, Y_dev = X_nv[dev_idxs,:], Y[dev_idxs, WHICH_Y]
X_test, Y_test= X_nv[test_idxs,:], Y[test_idxs, WHICH_Y]


# SKLEARN Models


# Benchmark errors
benchmark1 = np.median(np.abs(np.median(Y_train)- Y_test))
benchmark2 = np.mean(np.abs(np.median(Y_train)- Y_test))

#######
# LASSO
#######

# Hyperparameter grid for dev set
r_Lasso = -6 * np.random.rand(100)
hp_grid_Lasso = np.sort(10**r_Lasso)
tune_scores_Lasso = np.zeros(hp_grid_Lasso.shape[0])

# NOTE: If classifier for LASSO then need to use the 0.5 rule

# Tune
k=0
for alpha in hp_grid_Lasso:
	tune_Lasso = Lasso(alpha=alpha)
	# Fit
	tune_Lasso.fit(X_train, Y_train)
	# Get score on dev set
	if WHICH_Y==0:
		tune_scores_Lasso[k] = tune_Lasso.score(X_dev, Y_dev)
	else:
		tune_scores_Lasso[k] = f1_score(Y_dev, (1.0*(tune_Lasso.predict(X_dev)>=0.5)))
	k+=1

# Get best score on dev set
alpha_star = hp_grid_Lasso[np.where(tune_scores_Lasso==tune_scores_Lasso.max())][0]

# Now go to test set
model_Lasso = Lasso(alpha=alpha_star)
model_Lasso.fit(X_train, Y_train)

if WHICH_Y==0:
	#final_score_Lasso = model_Lasso.score(X_test, Y_test)
	#final_score_Lasso2 = np.mean(np.abs(model_Lasso.predict(X_test)- Y_test))
	final_score_Lasso = np.median(np.abs(model_Lasso.predict(X_test)- Y_test))
else:
	test_fits_Lasso = (1.0*(model_Lasso.predict(X_test)>=0.5))
	final_score_Lasso =  f1_score(Y_test, test_fits_Lasso)
	print('F1 Score Lasso: ' + str(final_score_Lasso))
	accuracy_score_Lasso = accuracy_score(Y_test, test_fits_Lasso)
	print('Accuracy Score Lasso: '+str(accuracy_score_Lasso))
	print(test_fits_Lasso[0:20])


# Get non0 coefficients for LASSO
non0coeff = np.where(model_Lasso.coef_!=0)

# 13th drive: +0.5 yards per play
# 2nd down: -0.6 yards per play
# minute of play in a quarter: 1.16 yards
# yardline (both regular) and to go predict: more yards to go => less yards
# score of both teams: if offensive team has higher score, then predict more yards
# if defensive team has higher score, then predict less yards


# Previously done: ADABoost
# #######
# # ADABoost
# #######

# r_ADA = -5 * np.random.rand(100)
# hp_grid_ADA = np.sort(10**r_ADA)
# tune_scores_ADA = np.zeros(hp_grid_ADA.shape[0])

# k=0
# if WHICH_Y==0:
# 	for learning_rate in hp_grid_ADA:
# 		tune_ADA = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth=2),
# 			learning_rate=learning_rate)
# 		tune_ADA.fit(X_train, Y_train[:, WHICH_Y])
# 		tune_scores_ADA = tune_ADA.score(X_dev, Y_dev[:, WHICH_Y])
# 		k+=1
# 	learning_rate_star = hp_grid_ADA[np.where(tune_scores_ADA==tune_scores_ADA.max())][0]
# 	model_ADA = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth=2), 
# 		learning_rate=learning_rate_star)
# 	model_ADA.fit(X_train, Y_train[:, WHICH_Y])
# 	final_score_ADA = model_ADA.score(X_test, Y_test[:, WHICH_Y])
# else:
# 	for learning_rate in hp_grid_ADA:
# 		tune_ADA = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2),
# 			learning_rate=learning_rate)
# 		tune_ADA.fit(X_train, Y_train[:, WHICH_Y])
# 		tune_scores_ADA = f1_score(Y_dev[:, WHICH_Y], tune_ADA.predict(X_dev))
# 		k+=1
# 	learning_rate_star = hp_grid_ADA[np.where(tune_scores_ADA==tune_scores_ADA.max())][0]
# 	model_ADA = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2),
# 		learning_rate=learning_rate_star)
# 	model_ADA.fit(X_train, Y_train[:, WHICH_Y])
# 	final_score_ADA = f1_score(Y_test[:, WHICH_Y], model_ADA.predict(X_test))



#######
# Random Forest
#######

hp_grid_RF_n = np.unique(np.array(np.logspace(0,2,15, dtype="int64")))
hp_grid_RF_d = np.array([1,3,5,10,20])
hp_grid_RF = np.array(list(product(hp_grid_RF_n, hp_grid_RF_d)))
tune_scores_RF = np.zeros(hp_grid_RF.shape[0])

if WHICH_Y==0:
	for j in range(hp_grid_RF.shape[0]):
		tune_RF = RandomForestRegressor(
			n_estimators = hp_grid_RF[j,0],
			max_depth=hp_grid_RF[j,1],
			criterion="mae")
		tune_RF.fit(X_train, Y_train)
		tune_scores_RF[j] = tune_RF.score(X_dev, Y_dev)
		print(j)
	hp_RF_star = hp_grid_RF[np.where(tune_scores_RF==tune_scores_RF.max())][0]
	model_RF = RandomForestRegressor(
		n_estimators = hp_RF_star[0],
			max_depth=hp_RF_star[1],
			criterion="mae")
	model_RF.fit(X_train, Y_train)
	#final_score_RF = model_RF.score(X_test, Y_test)
	#final_score_RF2 = np.mean(np.abs(model_RF.predict(X_test)- Y_test))
	final_score_RF = np.median(np.abs(model_RF.predict(X_test)- Y_test))
	final_score_RF2 = np.mean(np.abs(model_RF.predict(X_test)- Y_test))
else:
	for j in range(hp_grid_RF.shape[0]):
		tune_RF = RandomForestClassifier(n_estimators = hp_grid_RF[j,0],
			max_depth=hp_grid_RF[j,1])
		tune_RF.fit(X_train, Y_train)
		tune_scores_RF[j] = f1_score(Y_dev, tune_RF.predict(X_dev))
	hp_RF_star = hp_grid_RF[np.where(tune_scores_RF==tune_scores_RF.max())][0]
	model_RF = RandomForestClassifier(n_estimators = hp_RF_star[0],
			max_depth=hp_RF_star[0])
	model_RF.fit(X_train, Y_train)
	test_fits_RF = model_RF.predict(X_test)
	final_score_RF = f1_score(Y_test, test_fits_RF)
	print(hp_RF_star)
	print('F1 score RF: ' + str(final_score_RF))
	accuracy_score_RF = accuracy_score(Y_test, test_fits_RF)
	print('Accuracy Score RF: '+str(accuracy_score_RF))
	print(test_fits_RF[0:20])



