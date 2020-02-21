import numpy as np
import pandas as pd 
import datetime


# GOAL: Output numpy array with the features merged
# and the (y, x) separation clear

# PARAMETERS
# Starting for debugging and troubleshooting
START = 0
OUTPUT = "data/merged_full"
CHECK_MULTI = 0

# Load in the data
x_video = np.load('data/merged_video.npy', allow_pickle=True)
x_kaggle = pd.read_csv("data/nfl_play_by_play_2009-2016(v3).csv")

# Now go through each image and merge the data on the kaggle data
# Easier to do these processing in for loops than merging using pandas 
# data science features here because of numpy / non-standard format

# Build the final output array
merged_array = x_video

# Get the columns to merge on
merged_cols = ['Date', 'qtr', 'time', 'posteam', 'DefensiveTeam']
# Get distinguishing ocls
dist_cols = ['down', 'PlayAttempted', 'PlayType']
# Acceptable play types
acceptable_plays = ["Pass", "Run", "Sack"]

# Counter for number that have multiple rows to deal with
if CHECK_MULTI==1:
	k_multi=0

# Some manual changes I know I need to make
	# Messed up the times in the coverage sheet, they are fixed now
	# this will not harm if correct
x_video[382, 1][4] = datetime.time(0, 48)
x_video[742, 1][4] = datetime.time(13, 20)
x_video[778, 1][4] = datetime.time(7, 35)

for row in range(START, x_video.shape[0]):
	game_info = x_video[row, 1]
	team1 = game_info[0]
	date = str(game_info[2]).split()[0]
	time = ":".join(str(game_info[4]).split(':')[0:2])
	matches = x_kaggle.loc[(x_kaggle['Date']==date) & (x_kaggle['qtr']==game_info[3]) & (x_kaggle['time']==time) & ((x_kaggle['posteam'].apply(lambda x: x.lower() if type(x) == str else np.nan)==team1) | (x_kaggle['DefensiveTeam'].apply(lambda x: x.lower() if type(x) == str else np.nan)==team1))]
	# If the length of matches is 1
	print(str(row)+" / " + str(x_video.shape[0])) if row % 10 == 0 else None
	if len(matches)==1:
		merged_array[row, 1] = np.array(matches)
	else:
		if CHECK_MULTI==1:
			k_multi+=1
			print("multi-row num " + str(k_multi) + " at " + str(row) + ", matching on playtype")
		sub_matches = matches.loc[matches['PlayType'].isin(acceptable_plays)]
		if len(sub_matches)==1:
			merged_array[row, 1] = np.array(sub_matches)
		else:
			# Accidentally got 1 penalty play so need to change play type for that guy
			if row==997:
				sub_matches = matches.iloc[2:,]
				merged_array[row, 1] = np.array(sub_matches)
			else:
				raise AttributeError("No rows on " + str(row))

# Save output
print("DONE")
if OUTPUT:
	np.save(OUTPUT, merged_array)
	print("OUTPUT SAVED IN DATA")