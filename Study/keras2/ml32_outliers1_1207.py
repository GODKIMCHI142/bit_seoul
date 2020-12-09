import numpy as np

def outliers(data_out):
    quartile_1, quartile_3 = np.percentile(data_out, [25,75],interpolation="liner")
    print("1사분위 : ",quartile_1) # 3.25
    print("3사분위 : ",quartile_3) # 97.5
    iqru = quartile_3 - quartile_1 # 94.25
    iqrl = quartile_1 - quartile_1 # -94.25
    lower_bound = quartile_1 - (iqrl * 1.5) 
    upper_bound = quartile_3 + (iqru * 1.5)
    u_idx = np.where(data_out>upper_bound)
    l_idx = np.where(data_out<lower_bound)
    return  (u_idx,l_idx)
    
a = np.array([1,2,3,4,10000,6,7,5000,90,100])

b = outliers(a)
print("이상치의 위치 : ",b)
# 이상치의 위치 :  ((array([4, 7], dtype=int64),), (array([0, 1, 2], dtype=int64),))

# np.where() : 비교후 인덱스 반환
# np.percentile() : data의 interpolation으로 데이터를 자른 value를 반환
#                 : 값들을 정렬해준다. 
# liner : 백분위 
# nearest : liner의 가장 가까운 진짜 값
# lower : 백분위보다 낮은 값 중 제일큰 값 
# higher : 백분위보다 큰값 중 제일낮은 값
# midpoint : linear로 나온 앞뒤값의 평균값.
