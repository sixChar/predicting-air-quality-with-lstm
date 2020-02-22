# predicting-air-quality-with-lstm
Contains code for a simple lstm model (implemented using keras from tensorflows tf.keras module) to predict hourly Beijing air quality data.

The data is not included in this repository and can be found [here](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data).

If you want to make use of this repository then you will have to download the data and put the path to said data in at line 
10 in *data_utils.py*.

The file *lstm_model_weights.h5* is the weights of the model trained on somewhere around 60000 batchs, with a batch size of 8 
and sequence lengths between 128 and 256 steps. Likely not enough training but I don't have a gpu.

# Results
  The mean absolute error over time plots for most attributes being predicted can be seen in the _plots_ directory. The error is the average error from 128 sequence predictions. The x axis is in hours and the different lines are for different stations. All of these attributes were normalized with mean 0 and standard deviation 1 as part of the preprocessing so an error of 0.5 is about 0.5 standard deviations away from the actual value.

  The model cannot accurately predict most of the measured atrtibutes more than a few hours out and even then it's not great. 
The average first hour error is about 0.2 standard deviations which increases quite rapidly in the next few hours. However for most attributes, the error does seem to plateau at around 1 standard deviation which suggests the model isn't making wildley outrageous predictions. Also it seems from the error plots that the network is having predictable spikes in error (the TEMP plot is a good example). Since these spikes correlate across stations and in spite of the somewhat large (128) testing sample size it seems likely that it is a failure of the model to predict cycles in the data. For example the day/night temperature cycle in the TEMP data.

  On average the error is roughly similar or uncorrelated between stations, however this does not hold for the wind speed error in Gucheng which is significantly higher than the wind speed error for the rest of the stations which is interesting.

  I think that the model still has a lot of room to improve with more training. It still seemed to be improving but after 4-5 days I felt my computer had earned a break.

  In conclusion, don't try and use this to predict the pollution levels in Beijing if you have asthma.



