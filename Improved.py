from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from keras.models import Sequential
import keras
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler

# For Ignoring warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

file = pd.read_csv('xAPI-Edu-Data.csv', delimiter = ',', engine = 'python')
X = file.iloc[:, 5:13].values
y = file.iloc[:, -1].values

labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
X[:, 5] = labelencoder.fit_transform(X[:, 5])
X[:, 6] = labelencoder.fit_transform(X[:, 6])
X[:, 7] = labelencoder.fit_transform(X[:, 7])

y = labelencoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

model = Sequential()

model.add(Dense(16, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

model.add(Dense(32, kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, kernel_initializer='normal',activation='relu'))
model.add(Dense(32, kernel_initializer='normal',activation='relu'))

model.add(Dense(units = 1))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics = ['accuracy'])

model.summary()

checkpoint = ModelCheckpoint('./Weights/Weights-{epoch:03d}--{val_loss:.5f}.hdf5', monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

# Train the model
history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''

wights = './Weights/Weights-388--0.59433.hdf5' # choose the best checkpoint 
model.load_weights(wights)
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
test_prediction = model.predict(X_test).flatten()

plt.plot(loss_values)
plt.show()
'''
