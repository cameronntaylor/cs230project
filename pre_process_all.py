import numpy as np
import sklearn
from sklearn import preprocessing
import pandas as pd 


# OUTPUT pre-processed
X_V_OUTPUT = "data/preprocessed/X_v_pp"
X_NV_OUTPUT = "data/preprocessed/X_nv_pp"
Y_OUTPUT = "data/preprocessed/Y_pp"

# Read in data
data = np.load('data/merged_full.npy', allow_pickle=True)
kaggle_df_cols = np.array(pd.read_csv("data/nfl_play_by_play_2009-2016(v3).csv").columns)

# Separate out x's and y's by df columns
y_yds = "Yards.Gained"
y_playtype = "PlayType"
y_firstdown = "FirstDown"

y_yds_col = np.where(kaggle_df_cols==y_yds)[0][0]
y_play_col = np.where(kaggle_df_cols==y_playtype)[0][0]
y_firstdown_col = np.where(kaggle_df_cols==y_firstdown)[0][0]
valid_xs = ['Drive', 'qtr', 'down', 'time', 'yrdln', 'yrdline100', 'ydstogo', 
'posteam', 'DefensiveTeam', 'PosTeamScore', 'DefTeamScore']
X_cols = [i for i in range(len(kaggle_df_cols)) if (kaggle_df_cols[i] in valid_xs)]

# Want to transform some variables to one-hot encoding

# One hot encoding on variables (see model notes)
output_array = data

# Get all kaggle data
kaggle_array = np.stack(data[:,1][:][:]).reshape((1055,102))

# NOTE: Some leftover play types that I have are kickoffs or no play
# I SHOULD GET RID OF THOSE

kickoffs = np.where(kaggle_array[:,y_play_col]=="Kickoff")[0]
no_plays = np.where(kaggle_array[:,y_play_col]=="No Play")[0]
valid_idxs = np.setdiff1d(
	np.setdiff1d(np.array(range(kaggle_array.shape[0])), 
		kickoffs),
	no_plays)

kaggle_array = kaggle_array[valid_idxs, :]

# NOTE: Get the minute of the time 

kaggle_array[:, 5] = np.array([xi.split(":")[0] for xi in kaggle_array[:, 5]])

# One-hot encode cols
X_cols_one_hot = [X_cols[i] for i in range(len(X_cols)) if i in [0,1,2,3,7,8]]


# GET Xs
# Ex: one-hot encode drive #
enc = preprocessing.OneHotEncoder()
enc.fit(kaggle_array[:, X_cols_one_hot[0]].reshape(-1,1))
X_drive = enc.transform(kaggle_array[:, X_cols_one_hot[0]].reshape(-1,1)).toarray()
enc.fit(kaggle_array[:, X_cols_one_hot[1]].reshape(-1,1))
X_qtr = enc.transform(kaggle_array[:, X_cols_one_hot[1]].reshape(-1,1)).toarray()
enc.fit(kaggle_array[:, X_cols_one_hot[2]].reshape(-1,1))
X_down = enc.transform(kaggle_array[:, X_cols_one_hot[2]].reshape(-1,1)).toarray()
enc.fit(kaggle_array[:, X_cols_one_hot[3]].reshape(-1,1))
X_time = enc.transform(kaggle_array[:, X_cols_one_hot[3]].reshape(-1,1)).toarray()
enc.fit(kaggle_array[:, X_cols_one_hot[4]].reshape(-1,1))
X_offteam = enc.transform(kaggle_array[:, X_cols_one_hot[4]].reshape(-1,1)).toarray()
enc.fit(kaggle_array[:, X_cols_one_hot[5]].reshape(-1,1))
X_defteam = enc.transform(kaggle_array[:, X_cols_one_hot[5]].reshape(-1,1)).toarray()

# Other columnns
X_yrdline = kaggle_array[:, X_cols[4]].reshape(len(valid_idxs),1)
X_yrdline100 = kaggle_array[:, X_cols[5]].reshape(len(valid_idxs),1)
X_yrdstogo= kaggle_array[:, X_cols[6]].reshape(len(valid_idxs),1)
X_offteamscore = kaggle_array[:, X_cols[9]].reshape(len(valid_idxs),1)
X_defteamscore = kaggle_array[:, X_cols[10]].reshape(len(valid_idxs),1)

X_nonvideo = np.concatenate(
	(X_drive, 
		X_qtr, 
		X_down,
		X_time,
		X_yrdline,
		X_yrdline100,
		X_yrdstogo,
		X_offteam,
		X_defteam,
		X_offteamscore,
		X_defteamscore), 
	axis=1)


X_video = np.empty((len(valid_idxs), 230, 119, 3))
for row in range(X_video.shape[0]):
	X_video[row,:,:,:] = data[valid_idxs][row, 0].reshape(1,230,119,3)

# GET ys
y_real = kaggle_array[:, y_yds_col].reshape(-1,1)

y_posplay = (1.0*(kaggle_array[:, y_yds_col]>0)).reshape(-1,1)

y_ontrack = (1.0*(kaggle_array[:, y_yds_col]>3)).reshape(-1,1)

y_firstdown = (kaggle_array[:, y_firstdown_col]).reshape(-1,1)


# Put together

X = (X_video, X_nonvideo)
Y = np.concatenate((y_real, y_posplay, y_ontrack, y_firstdown), axis=1)

# Output
np.save(X_V_OUTPUT , X_video)
np.save(X_NV_OUTPUT , X_nonvideo)
np.save(Y_OUTPUT , y)
