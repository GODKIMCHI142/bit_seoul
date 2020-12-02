import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape,x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape,y_test.shape) # (60000,) (10000,)

from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)


x_train = x_train.reshape(60000,14,14,4).astype('float32')/255.
x_test  = x_test.reshape(10000,14,14,4).astype('float32')/255. 

# 2. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense, Dropout
from tensorflow.keras.layers import LSTM
# 모델을 함수로 만들어 보았다.
def build_model(drop=0.5, optimizer="adam"):
    inputs = Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3],),name="input")
    x = Conv2D(256,(3,3),activation="relu",name="hidden1")(inputs)
    x = Dropout(drop)(x)
    x = Conv2D(128,(3,3),activation="relu",name="hidden2")(x)
    x = Dropout(drop)(x)
    x = MaxPooling2D(pool_size=2,name="hidden3")(x)
    x = Flatten()(x)
    x = Dense(64,activation="relu",name="hidden4")(x)
    x = Dropout(drop)(x)    

    outputs = Dense(10,activation="softmax",name="output")(x)
    model1 = Model(inputs=inputs, outputs=outputs)
    model1.summary()
    model1.compile(optimizer=optimizer, metrics=["acc"],
                   loss="categorical_crossentropy")

    return model1


# 파라미터
def create_hyperparameter():
    # batches = [10, 20, 30, 40, 50]
    # optimizers = ['rmsprop', 'adam', 'adadelta']
    # dropout = np.linspace(0.1, 0.5, 5)
    batches = [50]
    optimizers = ['adam']
    # dropout = np.linspace(0.1,0.5, 5)
    return{"batch_size":batches, "optimizer":optimizers}


hyperparameters = create_hyperparameter()


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
model = KerasClassifier(build_fn=build_model, verbose=1)
# keras 모델을 sk_learn에서 사용하기 위해 형변환

from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
search = GridSearchCV(model,hyperparameters,cv=3,verbose=1)
search.fit(x_train,y_train)

print(search.best_params_) # 최적의 파라미터를 찾는다.
acc = search.score(x_test,y_test)
print("acc : ",acc)
# {'batch_size': 50, 'optimizer': 'adam'}
# 200/200 [==============================] - 0s 1ms/step - loss: 0.0786 - acc: 0.9768
# acc :  0.9768000245094299