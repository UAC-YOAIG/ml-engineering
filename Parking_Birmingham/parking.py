"""
Created on Wed Apr 24 22:42:29 2019

@author: yannick
"""

import pandas as pd

#Importing the dataset
dataset = pd.read_csv("dataset.csv")

#Feature Engineering
dataset["is_shopping_park"] = dataset.apply(
        lambda row: 1 if row.SystemCodeNumber=="Shopping" else 0, axis = 1)

#un_d = dict(dataset.groupby(['SystemCodeNumber']).count())
#make data structure for predictiion
#we will predict the occupancy in time t+1 with capacity,occupancy,
#is_shopping_park parameters at time t
structured_data = list()
for i,row in dataset.iterrows():
    if i+1<len(dataset):
        f_row = dataset.iloc[i+1]
        if row.SystemCodeNumber==f_row.SystemCodeNumber:
            structured_data.append([row.Capacity, row.Occupancy,
                                    row.is_shopping_park,f_row.Occupancy])

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
ss_data = sc.fit_transform(structured_data)

#split into train and test set
from sklearn.model_selection import train_test_split
X = ss_data[:,:-1]
y = ss_data[:,-1]
train_X, test_X, train_y, test_y= train_test_split(X, y, test_size = 0.2,
                                                   random_state = 0)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# design network
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(train_X, train_y, batch_size = 32, epochs = 5, shuffle = False)
predicted_y = model.predict(test_X)

#evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error
abs_err = mean_absolute_error(test_y, predicted_y)
rmse = mean_squared_error(test_y, predicted_y)**0.5