import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras.optimizers import Adam, RMSprop
from keras import metrics
from keras import callbacks
from keras import regularizers
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def read_and_clean_file(filename):
  data = pd.read_csv(filename)
  data = data.drop(["Unnamed: 0", "Name of show"], axis=1) #"Name of show" and "Episode" are same

  data['Start_time'] = data['Start_time'].str[11:]
  data["Start_time"] = (data["Start_time"].str[0:2].astype(float)) + (data["Start_time"].str[3:5].astype(float))/60
  data["Length"] = data["Length"]/4
  data = data.drop(["End_time"], axis=1)

  data["Year"] = data["Year"].astype(str)
  data["Month"] = data["Date"].str[5:7]
  data["Day"] = data["Date"].str[8:10].astype(np.int8)
  data = data.drop(["Date"], axis=1)

  data = data.drop(["Name of episode"], axis=1) #Not using these feature

  for col in ["First time or rerun", "# of episode in the season", "Movie?", "Game of the Canadiens during episode?"]: 
    data[col] = (data[col] == "Yes").astype(np.int8)

  return data
  
def get_column_encoder(data, col_name, vocab_size=1000):
  groups = pd.DataFrame(data[col_name]).groupby(col_name).groups
  count = list(map(lambda x: (x, len(groups[x])), groups))
  count = np.array(sorted(count, key = lambda x: x[1], reverse = True))
  encoder = {}
  encoder["OTHER"] = 0
  for i in range(vocab_size):
    encoder[count[i][0]] = i+1
  return encoder
  
def encode_col(encoder, data, col_name):
  for i in range(data.shape[0]):
    if data[col_name][i] in encoder:
      data.at[i, col_name] = encoder[data[col_name][i]]
    else:
      data.at[i, col_name] = encoder["OTHER"]
  return data
  
  
