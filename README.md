# C230: Deep Learning for In-Game NFL Predictions
## Cameron Taylor | Stanford GSB | Winter 2020

## Description of Code


### Data Processing
1. image_process.py = takes in directories of raw screenshots from plays and outputs numpy arrays of RGB
   - Key parameters = size of output numpy array for images
   - Key input = raw screenshots, coverage array for screenshots that contains time of play in game, teams
   - Output = numpy array of image data along with time of play in games, teams
  
2. merge_data.py = merge image data with Kaggle data on time of play in game, teams
   - Key parameters = output directory for merged data
   - Key input = numpy array from Kaggle, numpy array of image data
   - Output = merged numpy array
   
3. pre_process_all.py = one-hot encode kaggle features, clean data for usable plays, get labels
   - Key parameters = output directory for pre-processed data
   - Key input = merged numpy array
   - Output = X, Y

### Models

1. ml_features.py = benchmark models with Kaggle data
   - Key parameters = train/dev/test split, which y to predict
   - Key input = pre-processed non-image X, Y
   - Output = tuning parameters and model error

2. shallow_cnn.py / shallow_cnn_classify.py = shallow cnn model on image and non-image data (yards / play classification)
   - Key parameters = hyperparameters (epochs, learning rate, mini-batch, num hidden units, filter size, L2 parameter)
   - Key input = X, Y
   - Output = model error
 
3. transfer_learn.py / transfer_learn_classify.py = VGG19 transfer learning model on image and non-image data (yards / play classification)
   - Key parameters = hyperparameters (epochs, learning rate, mini-batch, num hidden units, num layers to take from VGG19, L2 parameter)
   - Key input = X, Y
   - Output = model error 

4. transfer_learn_classify_tune.py = VGG19 transfer learning model on image and non-image data w/ more systematic tuning performance and logging built in (yards / play classification)
   - Key parameters = hyperparameters (epochs, learning rate, mini-batch, num hidden units, num layers to take from VGG19, L2 parameter)
   - Key input = X, Y
   - Output = log of model error and hyperparameters 

