import numpy as np
import csv

import tensorflow as tf
from tensorflow import keras
from tensorflow import optimizers as tf_optimizers

from sklearn.model_selection import train_test_split


def load_training_data(train_path):
    """Loads in data to train a model

    Args:
      train_path:  The path for the file of our training data

    Returns:
      text_x: Features for our test data
      test_y: Targets for our test data
      train_x: Features for our train data
      train_y: targets for our train data

    """

    reader = csv.reader(open(train_path), delimiter=",")
    temp = list(reader)
    train_data = np.array(temp[1:]).astype("float")
    x = train_data[0:, 1:]
    y = train_data[0:, 0]
    x /= 255

    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=1337)

    train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
    test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)

    # This segment is used to adjust the values for each pixel to either 1 or 0.
    # I am doing this because the digit reader on my website is black and white instead
    # of having shades of grey in between.
    for i in range(train_x.shape[0]):
        for j in range(0, 28):
            for k in range(0, 28):

                if train_x[i][j][k] > 0.5:
                    train_x[i][j][k] = 1
                else:
                    train_x[i][j][k] = 0

    return test_x, test_y, train_x, train_y


def build_cnn():
    """Builds a convolutional neural net (cnn) using tensorflow

    Args:

    Returns:
      model: The model for the cnn

    """
    drop_out = 0.25

    model = keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=5, padding="valid", activation='relu'))
    model.add(tf.keras.layers.Dropout(drop_out))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.Conv2D(64, kernel_size=4, padding="valid", activation='relu'))
    model.add(tf.keras.layers.Dropout(drop_out))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, padding="valid", activation='relu'))
    model.add(tf.keras.layers.Dropout(drop_out))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer=tf_optimizers.Adam(learning_rate=8e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


train_path = "E:/digit-recognizer/train.csv"

test_x, test_y, train_x, train_y = load_training_data(train_path)

model = build_cnn()

model.fit(train_x, train_y, shuffle=True, batch_size=500, epochs=6, verbose=1)
test_loss, test_acc = model.evaluate(test_x, test_y)

model.save('MNIST-Reader.h5')

print("... \n")
print("-----------------------------------")
print("LOSS: ", test_loss)
print("ACCURACY: ",  test_acc)
print("-----------------------------------\n")



