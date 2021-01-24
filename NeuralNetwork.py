#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import tensorflow as tf
import pandas as pd
import seaborn as sns
from keras.layers import (Input, Concatenate, concatenate,
                          BatchNormalization, Conv2D, Flatten, Dense)
from keras.models import Model, load_model
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

print(tf.__version__)
tf.keras.backend.set_floatx('float64')


def getCheckPoint(filepath, monitor, verbose=1, mode='min'):
    return ModelCheckpoint(filepath,
                           monitor=monitor,
                           verbose=verbose,
                           save_best_only=True,
                           mode=mode)


def getEarlyStop(monitor, verbose=1, mode='min'):
    return EarlyStopping(monitor=monitor, mode='min', patience=5)


# # Machine learning using Tensorflow

# ## Import data

# In[2]:


# Import data
# Predict survial

Train = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
Test = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

# In[3]:


X_Train = Train.drop('survived', axis=1)
X_Test = Test.drop('survived', axis=1)

y_Train = Train[['survived']]
y_Test = Test[['survived']]

# ## EDA
#

# In[4]:


observing_columns = "survived"

sns.pairplot(Train, size=2.5, hue=observing_columns)

plt.savefig("SNS_pairplot.png")
plt.close()

col = "embark_town"
pd.DataFrame({"Total": Train.groupby(col).count()['survived'],
              "p(survived| feature)": Train.groupby(col).sum()['survived'] / Train.shape[1]})

# In[7]:


for col in Train.columns:
    if Test[col].dtype == object:
        print(pd.DataFrame({"Total": Train.groupby(col).count()['survived'],
                            "p(survived)": Train.groupby(col).sum()['survived'] / Train.shape[0]}))
        print('-' * 36)

# In[8]:


X_Test.drop(X_Test[X_Test['embark_town'] == 'unknown'].index)

# ## Preprocessing

# In[9]:


try:
    X_Train, X_Test = X_Train.drop('deck', axis=1), X_Test.drop('deck', axis=1)
except KeyError:
    pass

X_Train = X_Train.drop(X_Train[X_Train['embark_town'] == 'unknown'].index)
X_Test = X_Test.drop(X_Test[X_Test['embark_town'] == 'unknown'].index)

cat_cols = []  # Categories
for col in X_Train.columns:
    if X_Test[col].dtype == object:
        cat_cols.append(col)

# ## One hot encoder

# In[10]:


enc = OneHotEncoder(handle_unknown='ignore')

enc.fit(X_Train[cat_cols])
cat_array = enc.transform(X_Train[cat_cols]).toarray()
cat_df = pd.DataFrame(cat_array, columns=enc.get_feature_names(cat_cols), index=X_Train.index)

X_Train_pre = pd.concat([X_Train, cat_df], axis=1).drop(cat_cols, axis=1)

cat_array = enc.transform(X_Test[cat_cols]).toarray()
cat_df = pd.DataFrame(cat_array, columns=enc.get_feature_names(cat_cols), index=X_Test.index)

X_Test_pre = pd.concat([X_Test, cat_df], axis=1).drop(cat_cols, axis=1)

# In[11]:

y_Train = y_Train.loc[X_Train_pre.index]
y_Test = y_Test.loc[X_Test_pre.index]

# ## Feature Engineering for the Model

# In[13]:

if __name__ == "__main__":
    # Get get hyper param
    early_stop = getEarlyStop('val_loss')
    # checkpoint = getCheckPoint('Best_model_with_{val_loss:.5}.model', 'val_loss')

    checkpoint = getCheckPoint('Best_model.model', 'val_loss')
    InputLayer = Input(shape=(X_Train_pre.shape[1]))
    OutputLayer = Dense(1, activation='sigmoid')(InputLayer)

    model = Model(inputs=InputLayer, outputs=OutputLayer)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy())

    model.summary()

    # model.fit(X_Train_pre, y_Train, epochs=5)

    hist = model.fit(X_Train_pre, y_Train,
                     validation_data=(X_Test_pre, y_Test),  # Tuple only
                     epochs=100,
                     callbacks=[checkpoint, early_stop],
                     verbose=2)
    plt.plot(hist.history['loss'], label='loss')
    plt.plot(hist.history['val_loss'], label='val_loss')

    plt.legend()
    plt.savefig("Train history")
    plt.close()

# # In[32]:
#
#
# class Logistic(keras.layers.Layer):
#     def __init__(self, units=1, input_dim=X_Train_pre.shape[1], name='log'):
#         super().__init__(name=name)
#         self.w = self.add_weight(
#             name='w', shape=(input_dim, units), initializer="random_normal", trainable=True
#         )
#         self.b = self.add_weight(name='b', shape=(units,), initializer="zeros", trainable=True)
#
#     def call(self, inputs):
#         return tf.sigmoid(tf.matmul(inputs, self.w) + self.b)
#
#

#
#
# InputLayer = Input(shape=(X_Train_pre.shape[1]))
# OutputLayer = Logistic(1, X_Train_pre.shape[1])(InputLayer)
#
# model = Model(inputs=InputLayer, outputs=OutputLayer)
#
# model.compile(loss=tf.keras.losses.BinaryCrossentropy())
#
# model.summary()
#
# model.fit(X_Train_pre, y_Train, epochs=5)
#
# class MyModel(Model):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = Conv2D(32, 3, activation='relu')
#         # self.flatten = Flatten()
#         self.d1 = Logistic(1, X_Train_pre.shape[1], name='log')
#         # self.d1 = Dense(1, activation='sigmoid')
#
#     def call(self, InputLayer):
#         x = self.d1(InputLayer)
#         return x
#
#
# model = MyModel()
# model.compile(loss='binary_crossentropy')
# model.fit(X_Train_pre, y_Train, epochs=55)
# model.save("Neural_network")
#
#
# test = [layer for layer in model.layers if not layer.built]
#
# # In[62]:
#
#
# test[0].built = True
