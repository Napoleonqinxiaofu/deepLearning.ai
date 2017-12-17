# fileName: mnsit
# author: xiaofu.qin
# create at 2017/12/17
# description:

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from Mnist import Mnist

train_images = Mnist.extract_images("../mnist-data/train-images.idx3-ubyte")
train_labels = Mnist.extract_labels("../mnist-data/train-labels.idx1-ubyte")
train_labels = keras.utils.to_categorical(train_labels, num_classes=10)

test_images = Mnist.extract_images("../mnist-data/t10k-images.idx3-ubyte")
test_labels = Mnist.extract_labels("../mnist-data/t10k-labels.idx1-ubyte")
test_labels = keras.utils.to_categorical(test_labels, num_classes=10)

print("extracted images and labels")

# define the model
model = Sequential()

# add one dense layer
model.add(Dense(100, activation="relu", input_dim=784))
model.add(Dense(500, activation="relu"))
model.add(Dense(10, activation="softmax"))

# sgd = SGD(lr=0.02, momentum=0.95)
# loss = categorical_crossentropy()

model.compile(
    optimizer="sgd",
    loss="categorical_crossentropy",
    metrics=["acc"]
)

model.fit(train_images, train_labels, epochs=20, batch_size=1024)

score = model.evaluate(test_images, test_labels)

print("prediction:", score)

# save the model
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
#
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")
