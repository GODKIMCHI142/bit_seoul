import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.data_utils import Sequence
import autokeras as ak

x_train = np.load("./data/keras63_imageDataGenerator2_x_train.npy")
x_test  = np.load("./data/keras63_imageDataGenerator2_x_test.npy")
y_train = np.load("./data/keras63_imageDataGenerator2_y_train.npy")
y_test  = np.load("./data/keras63_imageDataGenerator2_y_test.npy")
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)



# Initialize the image classifier
clf = ak.ImageClassifier(
    overwrite=True,
    max_trials=2
)

# Feed the image classifier with training data.
clf.fit(x_train,y_train,epochs=50)

# Predict with the best model
predicted_y = clf.predict(x_test)
print(predicted_y)


# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))
# clf.summary()


# 설치방법
# pip install autokeras
# pip install git+https://github.com/keras-team/keras-tuner.git@1.0.2rc4



# 4/4 [==============================] - 0s 12ms/step - loss: 1.2824 - accuracy: 0.6333
# [1.2823704481124878, 0.6333333253860474]






