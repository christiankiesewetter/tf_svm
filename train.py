from svm import SVMModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data

def get_data(c=1):
    (X_train, y_train), (X_val , y_val) = load_data()
    y_train = np.expand_dims(y_train, axis=-1)
    y_val = np.expand_dims(y_val, axis=-1)
    y_train = np.where(y_train == c, 1., 0.).astype(np.float32)
    y_val = np.where(y_val == c, 1., 0.).astype(np.float32)
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 127 - 1.
    X_val = X_val.reshape(X_val.shape[0], -1).astype(np.float32) / 127 - 1.
    return X_train, y_train, X_val, y_val



if __name__ == '__main__':
    X_train, y_train, X_val, y_val = get_data()
    svm = SVMModel(
        dimensions = X_train.shape[1],
        n_landmarks = 10,
        C = 1)
    svm.compile(optimizer=tf.keras.optimizers.Adam(lr=0.5e-03))
    svm.fit(X_train, y_train, validation_data=(X_val, y_val), epochs = 5)
    example = np.where(y_train == 1.)[0][20]
    print(svm(X_train[example,:]).numpy(), y_train[example,:])
