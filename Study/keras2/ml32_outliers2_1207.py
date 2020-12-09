import numpy as np

def outliers(data_out):
    # data_out = data_out.reshape(data_out.shape[0]*data_out.shape[1])
    quartile_1, quartile_3 = np.percentile(data_out, [25,75],interpolation="linear")
    print("1사분위 : ",quartile_1) # 4.25
    print("3사분위 : ",quartile_3) # 100.75
    iqru = quartile_3 - quartile_1 
    iqrl = quartile_1 - quartile_1 
    lower_bound = quartile_1 - (iqrl * 1.5) 
    upper_bound = quartile_3 + (iqru * 1.5)
    u_idx = np.where(data_out>upper_bound)
    l_idx = np.where(data_out<lower_bound)
    return  (("upper : ",u_idx),("lower : ",l_idx))
    
a = np.array([[1,2,3,4,10000,6,7,5000,90,100],
             [2,3,4,5,10001,7,8,5001,91,101],
             [4,5,6,7,10003,9,10,5003,93,103]])
print(a.shape) # (3, 10)

a = a.transpose()
print(a.shape) # (10, 3)


# b = outliers(a)
# print("이상치의 위치 : ",b)
# 2차원
# 이상치의 위치 :  ((
# 'upper : ', (
#     array([0, 0, 1, 1, 2, 2], dtype=int64), 
#     array([4, 7, 4, 7, 4, 7], dtype=int64))), (
# 'lower : ', (
#     array([0, 0, 0, 0, 1, 1, 1, 2], dtype=int64), 
#     array([0, 1, 2, 3, 0, 1, 2, 0], dtype=int64))))


# 1차원
# 이상치의 위치 :  ((
# 'upper : ', (array([ 4,  7, 14, 17, 24, 27], dtype=int64),)), (
# 'lower : ', (array([ 0,  1,  2,  3, 10, 11, 12, 20], dtype=int64),)))



# np.where() : 비교후 인덱스 반환
# np.percentile() : data의 interpolation으로 데이터를 자른 value를 반환
#                 : 값들을 정렬해준다. 
# linear : 백분위 
# nearest : linear의 가장 가까운 값
# lower : 백분위보다 낮은 값 중 제일큰 값 
# higher : 백분위보다 큰값 중 제일낮은 값
# midpoint : linear로 나온 앞뒤값의 평균값.
