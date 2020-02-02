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

# <-------------------------------------------> #
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


def fill_NAs(data):
  data["Start_time"] = data["Start_time"].fillna(np.round(data["Start_time"].mean()*2)/2)

  Temp = pd.DataFrame(data.groupby( ["Year", "Month", "Day", "Start_time"])["Temperature in Montreal during episode"].mean())
  NanIndex = data["Temperature in Montreal during episode"][data["Temperature in Montreal during episode"].isnull()].index
  for i in NanIndex:
    d = data.loc[i]
    data.at[i, "Temperature in Montreal during episode"] = Temp.loc[d["Year"], d["Month"], d["Day"], d["Start_time"]].values[0]

  Temp = pd.DataFrame(data.groupby( ["Year", "Month", "Day"])["Temperature in Montreal during episode"].mean())
  NanIndex = data["Temperature in Montreal during episode"][data["Temperature in Montreal during episode"].isnull()].index
  for i in NanIndex:
    d = data.loc[i]
    data.at[i, "Temperature in Montreal during episode"] = Temp.loc[d["Year"], d["Month"], d["Day"]].values[0]
  return data

def add_missing_cols(ready_data, ready_test):
  missed_columns_in_test = ready_data.columns[ready_data.columns.isin(ready_test.columns) == False][:-1]
  for col in missed_columns_in_test:
    ready_test = ready_test.join(pd.Series(np.zeros(test.shape[0]), name=col).astype(int))
  cols = list(ready_data.columns)
  del(cols[-3])
  ready_test = ready_test[cols]
  return ready_test

# <---------------------------------------------------------->#
data_file_address = "drive/My Drive/data.csv"
test_file_address = "drive/My Drive/test.csv"

data = read_and_clean_file(data_file_address)
test = read_and_clean_file(test_file_address)

print("Count of NA in columns before filling NAs")
print(data.isnull().sum())

data = fill_NAs(data)
test = fill_NAs(test)

print("Count of NA in columns after filling NAs")
print(data.isnull().sum())

vocab_size = 3000
encoder = get_column_encoder(data, "Episode", vocab_size)

data = encode_col(encoder, data, "Episode")
test = encode_col(encoder, test, "Episode")


data["Episode"] = data["Episode"].astype(int)
test["Episode"] = test["Episode"].astype(int)

ready_data = pd.get_dummies(data[data.select_dtypes(object).columns], drop_first=True).join(data.select_dtypes(np.number))
ready_test = pd.get_dummies(test[test.select_dtypes(object).columns], drop_first=True).join(test.select_dtypes(np.number))

ready_test = add_missing_cols(ready_data, ready_test)

train_size = 500000

ready_data = ready_data.sample(len(ready_data))

train_data = ready_data[:train_size]
valid_data = ready_data[train_size:]

train_y = train_data["Market Share_total"].values
train_x_episode = train_data[ "Episode"].values
train_x = train_data.drop(["Market Share_total", "Episode"], axis=1).values

valid_y = valid_data["Market Share_total"].values
valid_x_episode = valid_data[ "Episode"].values
valid_x = valid_data.drop(["Market Share_total", "Episode"], axis=1).values

test_x_episode = ready_test[ "Episode"].values
test_x = ready_test.drop(["Episode"], axis=1).values


MU = []
Sigma = []
for i in range(train_x.shape[1]):
  MU.append(train_x.T[i].mean())
  Sigma.append(train_x.T[i].std())
  valid_x.T[i] = (valid_x.T[i] - MU[i])/Sigma[i]
  train_x.T[i] = (train_x.T[i] - MU[i])/Sigma[i]
  test_x.T[i]  = ( test_x.T[i] - MU[i])/Sigma[i]
MU.append(train_y.mean())
Sigma.append(train_y.std())
valid_y = (valid_y - train_y.mean())/train_y.std()
train_y = (train_y - train_y.mean())/train_y.std()

#<----------------------------------------------------------->#

m = train_y.mean()
print("MAE of Using Mean is ", np.mean(np.abs(m - valid_y)))


reg = LinearRegression().fit(train_x, train_y)
print("MAE of linear regression is ", np.mean(np.abs(reg.predict(valid_x) - valid_y)))


model1 = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(1,)),
  tf.keras.layers.Embedding(vocab_size+1, 16),
  tf.keras.layers.Flatten()
])

model2 = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(81,)),
  tf.keras.layers.Dense(120, activation='relu')
])

mergedOut = tf.keras.layers.Concatenate()([model1.output, model2.output])
mergedOut = tf.keras.layers.Flatten()(mergedOut)
mergedOut = tf.keras.layers.Dropout(0.2)(mergedOut)
mergedOut = tf.keras.layers.Dense(120, activation='relu')(mergedOut)
mergedOut = tf.keras.layers.Dropout(0.2)(mergedOut)
mergedOut = tf.keras.layers.Dense(1)(mergedOut)

model = tf.keras.models.Model([model1.input, model2.input], mergedOut)
                              
model.compile(loss='mae',
        optimizer="Adam",
        metrics=[metrics.mae])
model.summary()

epochs = 50
batch_size = 128
history = model.fit([train_x_episode, train_x], train_y,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=1, # Change it to 2, if wished to observe execution
    validation_data=([valid_x_episode, valid_x], valid_y))

y_pred = model.predict([valid_x_episode, valid_x])
print("MAE of NN is ", np.mean(np.abs(y_pred.flatten() - valid_y)))
print("R2 score of NN is ", r2_score(valid_y, y_pred.flatten()))



pred_file_address = "drive/My Drive/pred_test.csv"
y_pred_test = model.predict([test_x_episode, test_x])
pd.DataFrame(y_pred * Sigma[-1] + MU[-1]).to_csv(pred_file_address)
