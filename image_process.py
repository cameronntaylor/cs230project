import numpy as np
import pandas as pd 
from PIL import Image
#import opencv2
import os


# PARAMETERS
image_folder = "data/video/"
SIZE = (384, 216)
OUTPUT = "data/merged_video"
OUTPUT_IND = 0
SHOW_IMAGES = 0

# Image folders
	# In order from coverage file

game_folders = ["car_den_2016-09-08/", 
"cin_nyj_2016-09-08/",
"cle_ten_2016-10-16/",
"hou_jac_2016-11-13/",
"phi_chi_2016-09-19/",
"dal_was_2016-09-18/",
"ari_buf_2016-09-25/",
"no_nyg_2016-09-18/",
"pit_kc_2016-10-02/"]

# Read in coverage sheet
coverage = pd.read_excel("data/coverage.xlsx")
coverage = coverage.iloc[9:]

# Base array to output - fix it's size
# so that first dimension is an obs (m)
# and second dimension indexes across image data and merging data
if OUTPUT_IND==1:
	merged_array = np.empty((0,2))

# Loop by game
k0 = 0
for game in game_folders:
	# Get all images in that folder
	image_names = [f for f in os.listdir(image_folder+game) if os.path.isfile(os.path.join(image_folder+game, f))]
	# Need to sort by time taken (name order in folder)
	image_names.sort(key=lambda x: int(x.split()[4].replace('.','')))
	# Counter for images
	k=0
	# Get coverage slice that only has this game (relies on unique team 1)
	game_coverage = coverage[coverage['team1']==game.split("_")[0]]
	# Assert that they have the same lengths
	assert game_coverage.shape[0]==len(image_names)
	num_plays = len(image_names)
	# Print game name once start
	print(game)
	# Loop by image
	for image_name in image_names:
		# Get an image
		image = image_folder+game+image_name
		my_image = Image.open(image)
		# Resize / crop it according to the SIZE parameter
		# Cropping chosen so that in general cut out most stuff accept for the field (chosen with some case studies)
		my_imagers = my_image.resize(SIZE).crop((SIZE[0]*0.2, SIZE[1]*0.3, SIZE[0]*0.8, SIZE[1]*0.85))
		# Show image (randomly to see what they look like)
		if SHOW_IMAGES==1:
			my_imagers.show() if (np.random.uniform() < 0.0075) else None
			#my_imagers.show() if k0 in valid_idxs[[666, 251, 271, 44, 788]] else None
		# Get final size after cropping (230, 119) when SIZE = (384, 216)
		final_size = my_imagers.size
		
		# Now get pixel data
		pixels = my_imagers.getdata()
		# Put in numpy array: resized to be (width, height, channels(3))
		pixel_vals = np.array([(p[0], p[1], p[2]) for p in pixels]).reshape((final_size[0], final_size[1],3), 
			order="F")

		# Get game data information using the ordering of the plays
		team1 = game_coverage['team1'].iloc[k]
		team2 = game_coverage['team2'].iloc[k]
		date = game_coverage['date'].iloc[k]
		qtr = game_coverage['qtr'].iloc[k]
		time = game_coverage['time'].iloc[k]
		game_data_info = np.array([team1, team2, date, qtr, time])

		k0+=1

		# Store in dataframe
		# Shape goal = (num obs, 2)
		if OUTPUT_IND==1:
			merged_array = np.append(merged_array, [np.array((pixel_vals, game_data_info))], axis=0)
			# Confirm that shape is not changing on this dimension
			assert merged_array.shape[1] == 2

		# Add to counter and print for progress
		k+=1
		print(str(100*(k)/num_plays)) if k % 10 == 0 else None

# Save data to numpy file
if OUTPUT_IND==1:
	np.save(OUTPUT, merged_array)


