from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
# CNN (Convolutional Neural Network) 
# filters
# kernel_size
# strides
# padding
# 입력모양 : batch_size, rows, cols, channels
# input_shape = (rows, cols, channels)


# LSTM
# units
# return_sequence
# 입력모양 : batch_size, timesteps, feature
# input_shape = (timesteps, feature)

# conv2d param : (입력채널 x 필터폭 x 필터 높이 + bias) x 출력 채널수
# MaxPooling2D param : 0

model = Sequential()
model.add(Conv2D(10,(2,2),input_shape=(10,10,1))) # (9,9,10)
model.add(Conv2D(5,(2,2),padding='same'))         # (9,9,5)
model.add(Conv2D(3,(3,3),padding='valid'))        # (7,7,3)
model.add(Conv2D(7,(2,2)))                        # (6,6,7)
model.add(MaxPooling2D())                         # (3,3,7)
model.add(Flatten())                              # (63,) 3*3*7 = 63
model.add(Dense(1))                               # (1,)

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 9, 9, 10)          50
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 9, 9, 5)           205
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 7, 7, 3)           138
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 6, 6, 7)           91
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 3, 3, 7)           0
# _________________________________________________________________
# flatten (Flatten)            (None, 63)                0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 64
# =================================================================
# Total params: 548
# Trainable params: 548
# Non-trainable params: 0

model.summary()






