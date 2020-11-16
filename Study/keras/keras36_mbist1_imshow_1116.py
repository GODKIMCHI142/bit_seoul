import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist



(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape,x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape,y_test.shape) # (60000,) (10000,)
# print(x_train[0])
print(y_train[0])

plt.imshow(x_train[0])
plt.show()





from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder








































'''
# one hot encoding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text="가 나 다 라 마 바 사 아 자 차 카 타 파 하"
t = Tokenizer()
t.fit_on_texts([text])
print(t.word_index)
# {'가': 1, '나': 2, '다': 3, '라': 4, '마': 5, '바': 6, '사': 7, '아': 8, '자': 9, '차': 10, '카': 11, '타': 12, '파': 13, '하': 14}

sub_text="아 가 라 마 사 카 타 하"
encoded=t.texts_to_sequences([sub_text])
print(encoded)
# [[8, 1, 4, 5, 7, 11, 12, 14]]

one_hot = to_categorical(encoded)
print(one_hot)
# [[[0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.] = 인덱스 8의 원-핫 벡터
#   [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] = 인덱스 1의 원-핫 벡터 
#   [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] = 인덱스 4의 원-핫 벡터
#   [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.] = 인덱스 5의 원-핫 벡터
#   [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.] = 인덱스 7의 원-핫 벡터
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.] = 인덱스 11의 원-핫 벡터
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.] = 인덱스 12의 원-핫 벡터
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]] = 인덱스 14의 원-핫 벡터

text2="가 나 다 라 마 바 사 아 자 차 카 타 파 하"
t2 = Tokenizer()
t2.fit_on_texts(text2)
print(t2.word_index)
# {'가': 1, '나': 2, '다': 3, '라': 4, '마': 5, '바': 6, '사': 7, '아': 8, '자': 9, '차': 10, '카': 11, '타': 12, '파': 13, '하': 14}

sub_text2="아 가 라 마 사 카 타 하"
encoded2=t2.texts_to_sequences(sub_text2)
print(encoded2)
# [[8], [], [1], [], [4], [], [5], [], [7], [], [11], [], [12], [], [14]]
'''








