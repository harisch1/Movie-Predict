import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
import preprocess

def neural(X, Y, X_pred):

    model = keras.Sequential([
        keras.layers.Dense(7, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=tf.keras.metrics.RootMeanSquaredError())

    model.fit(X.astype('float32'), Y.astype('float32'), epochs=1000, verbose=0)

    pred_y = model.predict(X_pred.astype('float32'))

    return pred_y